import rospy
from typing import Optional, Tuple
from sensor_msgs.msg import CameraInfo
import numpy as np
import cv2
from segmentation import get_mask_center

def compute_surface_normals(depth_image, kernel_size=5):
    """
    Compute surface normals from a depth image.
    
    Args:
        depth_image: 2D numpy array containing depth values
        kernel_size: Size of the Sobel kernel for gradient computation
    
    Returns:
        raw_normals: Raw normal vectors (for potential further processing)
    """
    # Ensure depth image is float32
    depth = depth_image.astype(np.float32)

    # fill in areas where depth is 0
    depth = cv2.inpaint(depth, (depth == 0).astype(np.uint8), inpaintRadius=4, flags=cv2.INPAINT_NS)

    # bilaterally filter normals
    depth = cv2.bilateralFilter(depth, 8, 75, 75)
    depth = cv2.GaussianBlur(depth, (3, 3), 0)
    depth = cv2.bilateralFilter(depth, 8, 75, 75)
    
    # Calculate gradients using Sobel
    sobelx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=kernel_size)
    
    # Surface normal = (-dz/dx, -dz/dy, 1)
    normals = np.dstack((-sobelx, -sobely, np.ones_like(depth)))
    
    # Normalize vectors
    norm = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
    normals = normals / (norm + 1e-10)  # Add small epsilon to avoid division by zero

    # bilaterally filter normals
    normals = cv2.bilateralFilter(normals, 5, 75, 75)
    normals = cv2.GaussianBlur(normals, (3, 3), 0)
    normals = cv2.bilateralFilter(normals, 5, 75, 75)
    normals = cv2.GaussianBlur(normals, (3, 3), 0)
    
    # Convert normals to RGB image
    # Scale from [-1, 1] to [0, 255]
    normals_rgb = ((normals + 1) * 127.5).astype(np.uint8)
    
    return normals_rgb, normals


def get_mask_point(mask: np.ndarray, depth_image: np.ndarray, color_info: CameraInfo) -> Optional[Tuple[float, float, float]]:
    """
    Calculate the point at the most prominent depth of the mask.

    Args:
        mask: 2D binary mask
        depth_image: 2D depth image
        color_info: camera calibration information
    """

    # mask depth
    mask_depth = np.multiply(mask, depth_image)

    # get average of non-zero values
    mask_depth = mask_depth[mask_depth != 0]
    if mask_depth.size == 0:
        return None
    
    avg_depth = np.mean(mask_depth)

    # only use depth value which are smaller (closer) than average
    mask_depth = mask_depth[mask_depth < avg_depth]
    avg_depth = np.mean(mask_depth)

    # get the center of the mask
    mask_center = get_mask_center(mask)
    print(mask_center)

    return calc_pos_by_depth(mask_center, avg_depth, color_info)

def calc_pos_by_depth(
    at_pixel: Tuple[int, int],
    depth: float,
    color_info: CameraInfo
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate 3D position from pixel coordinates and depth.
    
    Args:
        at_pixel: (x, y) pixel coordinates
        depth: depth value in meters
        color_info: camera calibration information
        
    Returns:
        3D point (x, y, z) in meters or None if invalid
    """
    if depth <= 0.01:
        return None
    
    # mm to m
    depth /= 1000.0

    # Extract intrinsic parameters
    fx = color_info.K[0]  # focal length x
    fy = color_info.K[4]  # focal length y
    cx = color_info.K[2]  # principal point x
    cy = color_info.K[5]  # principal point y

    # Calculate 3D coordinates
    x = (at_pixel[0] - cx) * depth / fx
    y = (at_pixel[1] - cy) * depth / fy
    z = depth

    return (x, y, z)

