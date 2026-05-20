#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rospy
from depth_processing import *
from segmentation import *
from proto_pub import *
from camera import CameraStream
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Pose, Point
import numpy as np
import cv2
from sympy import Plane, Point3D
import threading

rospy.init_node('pick_place', anonymous=True)
from transform import *
from arm_cmd import *
from scipy.spatial.transform import Rotation as _Rotation
from grasp_net import get_best_grasp, get_all_grasps, load_model

# ── Grasp geometry tuning ──────────────────────────────────────────────────────
# Contact-GraspNet was trained with a Panda gripper whose fingertips sit
# ~8.5 cm ahead of the wrist along the approach axis.  The predicted grasp
# position is therefore the WRIST position that places Panda fingers on the
# object.  The Kinova Robotiq fingers are shorter, so we push the wrist
# forward by GRASP_DEPTH_OFFSET_M to compensate.  Increase this value if
# the arm still stops short; decrease it if it pushes too hard into the object.
GRASP_DEPTH_OFFSET_M = 0.095   # metres – tunable

# Stand-off distance for Stage-1 (pre-grasp approach).
# Must be > GRASP_DEPTH_OFFSET_M so pre-grasp is always behind the object.
GRASP_STANDOFF_M = 0.15       # metres – tunable

# output publishers
image_pub = rospy.Publisher('pick_place_cam', Image, queue_size=1)
image_compressed_pub = rospy.Publisher('pick_place_cam/compressed', CompressedImage, queue_size=1)
pick_ready_pub = rospy.Publisher('pick_ready', Bool, queue_size=1)
pick_running_pub = rospy.Publisher('pick_running', Bool, queue_size=1)

window_name = 'Pick Place'

cam = CameraStream(
    color_topic='/camera/color/image_raw',
    depth_topic='/camera/depth_registered/sw_registered/image_rect_raw',
    color_info_topic='/camera/color/camera_info',
    depth_info_topic='/camera/depth/camera_info',
    use_compressed=True
)

# mouse click is handled in frame_cb
mouse_click = False
mouse_click_point = None
def mouse_cb(event, x, y, flags, param):
    global mouse_click
    global mouse_click_point
    if event == cv2.EVENT_LBUTTONUP:
        mouse_click = True
        mouse_click_point = (x, y)

mask = None
color = None
depth = None
color_info = None
depth_info = None

executing_pick = False

def get_frame_center():
    if color is None:
        return (0, 0)
    return (color.shape[1] // 2, color.shape[0] // 2)

def execute_pick():
    global mask, color, depth, color_info, depth_info, executing_pick

    if executing_pick:
        return

    executing_pick = True
    pick_ready_pub.publish(False)
    pick_running_pub.publish(True)

    starting_tool_position = get_frame_position("tool_frame", "base_link")
    starting_tool_rotation = get_frame_rotation_euler("tool_frame", "base_link")

    # ── 1. Get all grasp candidates from Contact-GraspNet ────────────────────
    grasps_cam, scores_cam = get_all_grasps(mask, depth, color_info)

    if grasps_cam is None:
        print('[execute_pick] No grasps found – aborting.')
        executing_pick = False
        return

    # ── 2. Get TF rotation camera_color_frame → base_link as numpy 3x3 ───────
    tf_cam_base = transform_frames("camera_color_frame", "base_link")
    if tf_cam_base is None:
        print('[execute_pick] TF lookup failed – aborting.')
        executing_pick = False
        return
    q = tf_cam_base.transform.rotation
    R_cam_base = _Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    # ── 3. Select best grasp by confidence, filtering impossible directions ───
    # We use pure confidence (Contact-GraspNet's own scoring) as the criterion.
    # The 3-stage approach (position → rotate → advance) already handles large
    # orientation changes, so there is no reason to bias toward the current tool
    # orientation – doing so was causing systematic top-down selection.
    #
    # We only filter out grasps that approach from BELOW the table (+Z in
    # base_link), which the arm physically cannot execute.
    best_idx = None
    for i, (g_cam, sc) in enumerate(zip(grasps_cam, scores_cam)):
        approach_dir_base = R_cam_base @ g_cam[:3, 2]
        # approach_dir_base[2] > 0.5 means the gripper would need to travel
        # upward to reach the object (approaching from below) – skip these.
        if approach_dir_base[2] > 0.5:
            continue
        best_idx = i
        break  # scores are already sorted descending – first valid = best

    if best_idx is None:
        # All grasps pointed upward (very unusual) – fall back to index 0
        best_idx = 0
        print('[execute_pick] Warning: all grasps have upward approach, using best by score.')

    best_grasp = grasps_cam[best_idx]
    approach_dir_base = R_cam_base @ best_grasp[:3, 2]
    print(f'[execute_pick] Selected grasp idx={best_idx}  '
          f'confidence={scores_cam[best_idx]:.3f}  '
          f'approach_base=[{approach_dir_base[0]:.2f}, {approach_dir_base[1]:.2f}, {approach_dir_base[2]:.2f}]  '
          f'pos_cam=[{best_grasp[0,3]:.3f}, {best_grasp[1,3]:.3f}, {best_grasp[2,3]:.3f}]')

    # ── 5. Transform grasp pose to base_link ──────────────────────────────────
    target_position = transform_pypoint(
        tuple(best_grasp[:3, 3].tolist()), "camera_color_frame", "base_link"
    )
    euler_cam = _Rotation.from_matrix(best_grasp[:3, :3]).as_euler('xyz')
    target_orientation = transform_pyrotation(
        Point3D(*euler_cam.tolist()), "camera_color_frame", "base_link"
    )

    if target_position is None or target_orientation is None:
        print('[execute_pick] TF transform failed – aborting.')
        executing_pick = False
        return

    # Pre-grasp position: GRASP_STANDOFF_M back from the corrected grasp position
    # (approach_dir_base already computed in grasp selection step above)
    # Final grasp position: push wrist forward to compensate for Kinova finger
    # length being shorter than the Panda fingers the model was trained with.
    grasp_position = Point3D(
        float(target_position.x) + approach_dir_base[0] * GRASP_DEPTH_OFFSET_M,
        float(target_position.y) + approach_dir_base[1] * GRASP_DEPTH_OFFSET_M,
        float(target_position.z) + approach_dir_base[2] * GRASP_DEPTH_OFFSET_M,
    )
    pre_target_position = Point3D(
        float(target_position.x) - approach_dir_base[0] * GRASP_STANDOFF_M,
        float(target_position.y) - approach_dir_base[1] * GRASP_STANDOFF_M,
        float(target_position.z) - approach_dir_base[2] * GRASP_STANDOFF_M,
    )

    # ── 6. Open gripper ───────────────────────────────────────────────────────
    grip(0)

    # ── Stage 1: position-only pre-grasp, keep current orientation ────────────
    # Moving position while keeping the current orientation maximises IK
    # success because the arm only needs to solve for XYZ, not for a full
    # reorientation at the same time.
    arm_set_position(pre_target_position)
    print('[execute_pick] Stage 1: pre-grasp position reached')

    # ── Closed-loop refinement ────────────────────────────────────────────────
    # The camera is wrist-mounted, so after Stage 1 the camera pose has changed
    # and the object is now closer (better depth accuracy).  Re-run the full
    # grasp estimator from this new vantage point and update the grasp target
    # before committing to rotation and final advance.
    rospy.sleep(0.3)  # let arm settle and receive a fresh camera frame

    # Re-fetch camera→base rotation since the camera has moved with the arm.
    _tf2 = transform_frames("camera_color_frame", "base_link")
    if _tf2 is not None:
        _q2 = _tf2.transform.rotation
        R_cam_base = _Rotation.from_quat([_q2.x, _q2.y, _q2.z, _q2.w]).as_matrix()

    # Project the known object position (base_link) into the new camera frame
    # so we know where to click for re-segmentation.
    _obj_in_cam = transform_pypoint(
        (float(target_position.x), float(target_position.y), float(target_position.z)),
        "base_link", "camera_color_frame"
    )
    _ref_color = color
    _ref_depth = depth
    _ref_info  = color_info
    if (_obj_in_cam is not None and _obj_in_cam.z > 0
            and _ref_color is not None and _ref_depth is not None):
        _fx = _ref_info.K[0]; _cx = _ref_info.K[2]
        _fy = _ref_info.K[4]; _cy = _ref_info.K[5]
        _u = int(_fx * _obj_in_cam.x / _obj_in_cam.z + _cx)
        _v = int(_fy * _obj_in_cam.y / _obj_in_cam.z + _cy)
        _h, _w = _ref_color.shape[:2]
        if 0 <= _u < _w and 0 <= _v < _h:
            _new_mask = segment_image((_u, _v), _ref_color)
            _new_grasps, _new_scores = get_all_grasps(_new_mask, _ref_depth, _ref_info)
            if _new_grasps is not None:
                _ridx = None
                for _i, _g in enumerate(_new_grasps):
                    if (R_cam_base @ _g[:3, 2])[2] > 0.5:
                        continue
                    _ridx = _i
                    break
                if _ridx is None:
                    _ridx = 0
                _rg = _new_grasps[_ridx]
                _ra = R_cam_base @ _rg[:3, 2]
                _rp = transform_pypoint(tuple(_rg[:3, 3].tolist()), "camera_color_frame", "base_link")
                _re = _Rotation.from_matrix(_rg[:3, :3]).as_euler('xyz')
                _ro = transform_pyrotation(Point3D(*_re.tolist()), "camera_color_frame", "base_link")
                if _rp is not None and _ro is not None:
                    target_position    = _rp
                    target_orientation = _ro
                    approach_dir_base  = _ra
                    grasp_position = Point3D(
                        float(_rp.x) + _ra[0] * GRASP_DEPTH_OFFSET_M,
                        float(_rp.y) + _ra[1] * GRASP_DEPTH_OFFSET_M,
                        float(_rp.z) + _ra[2] * GRASP_DEPTH_OFFSET_M,
                    )
                    print(f'[execute_pick] Refinement updated grasp: '
                          f'confidence={_new_scores[_ridx]:.3f}  '
                          f'approach=[{_ra[0]:.2f},{_ra[1]:.2f},{_ra[2]:.2f}]')
                else:
                    print('[execute_pick] Refinement: TF failed, keeping original grasp.')
            else:
                print('[execute_pick] Refinement: no grasps found, keeping original.')
        else:
            print('[execute_pick] Refinement: projected point out of frame, keeping original.')
    else:
        print('[execute_pick] Refinement: skipped (no frame or object behind camera).')

    # ── Stage 2: rotate to grasp orientation in place ─────────────────────────
    # Now that we are at the right spatial location the shoulder/elbow
    # configuration is already correct and a pure wrist rotation is enough.
    arm_set_rotation(target_orientation)
    print('[execute_pick] Stage 2: grasp orientation set')

    # ── Stage 3: advance along approach axis to final grasp position ──────────
    arm_set_position(grasp_position)
    print('[execute_pick] Stage 3: at grasp position')

    # ── Stage 4: close gripper ────────────────────────────────────────────────
    grip(1)

    # ── Stage 5: return to starting pose ─────────────────────────────────────
    arm_set_pose(starting_tool_position, starting_tool_rotation)
    print('[execute_pick] Returned to start')

    mask = None
    mouse_click_point = None   # clear the blue dot overlay on the camera feed
    pick_running_pub.publish(False)
    executing_pick = False


def frame_cb(_color, _depth, _color_info, _depth_info):
    global mouse_click, mask, mouse_click_point, point, color, depth, color_info, depth_info

    color = _color
    depth = _depth
    color_info = _color_info
    depth_info = _depth_info

    annotated = color
    if (mask is not None):
        annotated = draw_largest_contour_outline(mask, color)
    if (mouse_click_point is not None):
        annotated = draw_point(annotated, mouse_click_point, (0, 255, 255))

    cv2.imshow(window_name, annotated)
    image_pub.publish(cam.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))


    # resize annotated image by half
    annotated = cv2.resize(annotated, (0, 0), fx=0.5, fy=0.5)

    # Publish compressed version for WebSocket streaming
    compressed_msg = CompressedImage()
    compressed_msg.header.stamp = rospy.Time.now()
    compressed_msg.format = "jpeg"
    compressed_msg.data = np.array(cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 30])[1]).tobytes()
    image_compressed_pub.publish(compressed_msg)
    cv2.setMouseCallback(window_name, mouse_cb)

    key_code = cv2.waitKey(1)

    if key_code == ord('q'):
        cam.running = False
        cv2.destroyAllWindows()

    if (not executing_pick):
        if mouse_click:
            print('Mouse clicked')
            mouse_click = False
            mask = segment_image(mouse_click_point, color)
            point = get_mask_point(mask, depth, color_info)

            publish("closest_point", tuple_point_to_msg(point, "camera_color_frame"))

            pick_ready_pub.publish(True)

        if key_code == ord('p') and mask is not None:
            threading.Thread(target=execute_pick).start()
    elif mouse_click:
        mouse_click = False

# Pre-load Contact-GraspNet weights onto the GPU so the first pick has no lag
load_model()

cam.add_frame_callback(frame_cb)

def click_point_cb(msg: Point):
    global mouse_click_point, mouse_click

    print('Received click point')
    print(msg)
    if msg.x < 0 or msg.y < 0:
        return

    # Scale by 2x because compressed image is published at half resolution
    click_x = int(msg.x * 2)
    click_y = int(msg.y * 2)

    if (click_x >= cam.current_color_frame.shape[1] or click_y >= cam.current_color_frame.shape[0]):
        return

    mouse_click_point = (click_x, click_y)
    mouse_click = True

# input subscribers
click_point_sub = rospy.Subscriber('pick_click_point', Point, click_point_cb)
run_pick_sub = rospy.Subscriber('run_pick', Bool, lambda msg: threading.Thread(target=execute_pick).start())

cam.run()

while not rospy.is_shutdown() and cam.running:
    rospy.sleep(1)