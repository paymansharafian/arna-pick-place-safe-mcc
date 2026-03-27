#!/usr/bin/env python3
"""
Contact-GraspNet integration for the ARNA pick-and-place pipeline.

Wraps GraspEstimator to take a FastSAM binary mask + depth image + ROS CameraInfo
and return the best 6-DOF grasp pose (position + rotation matrix) in the camera frame.

Usage (called from main.py):
    from grasp_net import get_best_grasp, load_model

    load_model()                                  # call once at startup to avoid first-call lag
    position, rotation = get_best_grasp(mask, depth, color_info)
    # position: np.ndarray shape (3,) in camera_color_frame, metres
    # rotation: np.ndarray shape (3,3) SO3 rotation matrix in camera_color_frame
"""

import os
import sys
import numpy as np
import torch

from sensor_msgs.msg import CameraInfo

# ── locate checkpoint relative to this script ──────────────────────────────────
# scripts/ -> pick_place/ -> src/ -> ros/ -> home/legion-arna/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CKPT_DIR = os.path.realpath(
    os.path.join(_SCRIPT_DIR, '..', '..', '..', '..',
                 'contact_graspnet_pytorch', 'checkpoints', 'contact_graspnet')
)

# make sure the package src tree is on sys.path (it's installed editable, so
# normally not needed, but being explicit avoids any ROS env edge-cases)
_PKG_ROOT = os.path.realpath(os.path.join(_CKPT_DIR, '..', '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch.checkpoints import CheckpointIO

# ── singleton ──────────────────────────────────────────────────────────────────
_grasp_estimator: GraspEstimator = None


def load_model() -> None:
    """
    Load Contact-GraspNet weights onto the GPU (call once at node startup).
    Safe to call multiple times – subsequent calls are no-ops.
    """
    global _grasp_estimator
    if _grasp_estimator is not None:
        return

    if not os.path.isdir(_CKPT_DIR):
        raise FileNotFoundError(
            f'[grasp_net] Checkpoint directory not found: {_CKPT_DIR}'
        )

    global_config = config_utils.load_config(_CKPT_DIR, batch_size=1, arg_configs=[])
    estimator = GraspEstimator(global_config)
    estimator.model.eval()

    ckpt_io = CheckpointIO(
        checkpoint_dir=os.path.join(_CKPT_DIR, 'checkpoints'),
        model=estimator.model,
    )
    ckpt_io.load('model.pt')

    _grasp_estimator = estimator
    print(f'[grasp_net] Contact-GraspNet loaded from {_CKPT_DIR}')

    if torch.cuda.is_available():
        used = torch.cuda.memory_reserved() / 1e6
        total = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f'[grasp_net] VRAM after load: {used:.0f} / {total:.0f} MB')


# ── public API ─────────────────────────────────────────────────────────────────

def camera_info_to_K(color_info: CameraInfo) -> np.ndarray:
    """Convert a ROS CameraInfo message to a 3x3 camera intrinsics matrix (float64)."""
    return np.array(color_info.K, dtype=np.float64).reshape(3, 3)


def _run_inference(
    mask: np.ndarray,
    depth: np.ndarray,
    color_info: CameraInfo,
    z_range: list = None,
):
    """
    Internal: run point cloud extraction + Contact-GraspNet inference.

    Returns (grasps, scores) as parallel numpy arrays sorted by score descending:
        grasps  – np.ndarray (N, 4, 4)  SE3 matrices in camera_color_frame
        scores  – np.ndarray (N,)       confidence scores
    Returns (None, None) on failure.
    """
    if z_range is None:
        z_range = [0.15, 1.8]

    load_model()

    depth_m = depth.astype(np.float32) / 1000.0
    segmap  = mask.astype(np.int32)
    K       = camera_info_to_K(color_info)

    pc_full, pc_segments, _ = _grasp_estimator.extract_point_clouds(
        depth_m, K, segmap=segmap, rgb=None, z_range=z_range, segmap_id=1,
    )

    if pc_full is None or len(pc_full) == 0:
        print('[grasp_net] Empty full point cloud.')
        return None, None

    if 1 not in pc_segments or len(pc_segments[1]) == 0:
        print('[grasp_net] No object points in segment.')
        return None, None

    print(f'[grasp_net] pc_full: {len(pc_full)} pts, '
          f'pc_segment: {len(pc_segments[1])} pts')

    with torch.no_grad():
        pred_grasps_cam, pred_scores, _, _ = _grasp_estimator.predict_scene_grasps(
            pc_full,
            pc_segments={1: pc_segments[1]},
            local_regions=True,
            filter_grasps=True,
            forward_passes=1,
        )

    # Collect results — after local_regions+filter_grasps results live under key 1
    grasps_list, scores_list = [], []
    for k in pred_grasps_cam:
        if np.any(pred_grasps_cam[k]) and len(pred_grasps_cam[k]) > 0:
            grasps_list.append(pred_grasps_cam[k])
            scores_list.append(pred_scores[k])

    if not grasps_list:
        print('[grasp_net] No grasps survived filtering.')
        return None, None

    grasps = np.concatenate(grasps_list, axis=0)  # (N, 4, 4)
    scores = np.concatenate(scores_list, axis=0)  # (N,)

    # Sort by score descending
    order  = np.argsort(scores)[::-1]
    grasps = grasps[order]
    scores = scores[order]

    print(f'[grasp_net] {len(grasps)} grasps  '
          f'best score={scores[0]:.3f}  '
          f'worst score={scores[-1]:.3f}')
    return grasps, scores


def get_all_grasps(
    mask: np.ndarray,
    depth: np.ndarray,
    color_info: CameraInfo,
    z_range: list = None,
):
    """
    Return ALL predicted grasp candidates, sorted by confidence (best first).

    Returns:
        (grasps, scores)
            grasps  – np.ndarray (N, 4, 4)  SE3 in camera_color_frame
            scores  – np.ndarray (N,)
        or (None, None) on failure.
    """
    return _run_inference(mask, depth, color_info, z_range)


def get_best_grasp(
    mask: np.ndarray,
    depth: np.ndarray,
    color_info: CameraInfo,
    z_range: list = None,
):
    """
    Return the single highest-confidence grasp.

    Returns:
        (position, rotation_matrix) in camera_color_frame, or (None, None).
            position         – np.ndarray (3,)
            rotation_matrix  – np.ndarray (3, 3)
    """
    grasps, scores = _run_inference(mask, depth, color_info, z_range)
    if grasps is None:
        return None, None
    best = grasps[0]
    print(f'[grasp_net] Best grasp  score={scores[0]:.3f}  '
          f'pos=[{best[0,3]:.3f}, {best[1,3]:.3f}, {best[2,3]:.3f}]')
    return best[:3, 3].copy(), best[:3, :3].copy()
