import cv2
import numpy as np
from typing import Optional, Tuple
import math
from ultralytics.models.fastsam import FastSAM
from typing import Optional, Tuple

predictor = FastSAM("FastSAM-s.pt")

def get_mask_center(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Calculate the center of mass of the segmentation mask."""
    if mask is None:
        return None
        
    # Calculate moments of the binary mask
    moments = cv2.moments(mask)
    
    # Check if the mask is not empty
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        return (center_x, center_y)
    return None

def fill_largest_contour(mask):
    """
    Fill a binary mask by finding and filling the largest contour.
    
    Parameters:
    mask (numpy.ndarray): Binary mask image as a numpy array
    
    Returns:
    numpy.ndarray: The processed binary mask with the largest contour filled
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty mask
    filled_mask = np.zeros_like(mask)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fill the largest contour
        cv2.drawContours(filled_mask, [largest_contour], -1, 1, -1)
    
    return filled_mask


def draw_largest_contour_outline(mask: np.ndarray, color_image: np.ndarray) -> np.ndarray:
    """
    Draw the outline of the largest contour on the color image.
    Args:
        mask: binary mask of the segmentation
        color_image: color image to draw on
    Returns:
        Color image with the contour outline drawn
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the color image
    image_with_contour = color_image.copy()
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw the contour on the image
        cv2.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), 2)
    
    return image_with_contour

def draw_point(image: np.ndarray, point: Tuple[int, int], color: Tuple[int, int, int]) -> np.ndarray:
    """
    Draw a point on the image.
    Args:
        image: color image to draw on
        point: (x, y) pixel coordinates
        color: (r, g, b) color tuple
    Returns:
        Color image with the point drawn
    """
    image_with_point = image.copy()
    cv2.circle(image_with_point, point, 5, color, -1)
    return image_with_point

def segment_image(seg_point: Tuple[int, int], color_image: np.ndarray) -> np.ndarray:
    """
    Segment the image at the given point.
    Args:
        seg_point: (x, y) pixel coordinates
        color_image: color image to segment
    Returns:
        A binary mask of the segmentation
    """

    # Get prediction from FastSAM using the click point
    results = predictor(color_image, points=[seg_point])
    
    # create blank mask
    mask = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)
    
    # Extract mask from the first detection
    if len(results) > 0 and results[0].masks is not None:
        # Get the binary mask array
        mask = results[0].masks.data[0].cpu().numpy()

    # resize
    mask = cv2.resize(mask, (color_image.shape[1], color_image.shape[0]))

    return smooth_mask(fill_largest_contour(mask), 5)

def get_mask_rotation(mask: np.ndarray) -> float:
    """
    Find the rotation angle of the mask.
    Returns angle between -90 and 90 degrees where:
    - 0 degrees represents vertical orientation
    - Positive angles (0 to 90) represent clockwise rotation from vertical
    - Negative angles (-90 to 0) represent counterclockwise rotation from vertical
    """
    if mask is None:
        return 0.0

    # Find contours and get the largest one
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
        
    contour = max(contours, key=cv2.contourArea)

    # Fit a minimum area rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Get center, width, height, and angle from rect
    (cx, cy), (width, height), opencv_angle = rect
    
    # Determine if width or height is longer to find primary axis
    is_width_longer = width > height
    
    # Get points along longer side
    if is_width_longer:
        pt1 = box[0]  # First point of longer side
        pt2 = box[1]  # Second point of longer side
    else:
        pt1 = box[1]  # First point of longer side
        pt2 = box[2]  # Second point of longer side
    
    # Calculate angle from vertical
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    angle_rad = math.atan2(dx, dy)  # No negative dy this time
    angle_deg = math.degrees(angle_rad)
    
    # Adjust to make vertical 0 degrees and normalize to -90 to 90 range
    angle_deg = angle_deg - 90  # Subtract 90 to make vertical 0 degrees
    
    # Normalize to -90 to 90 range
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
        
    # Invert angle to match clockwise = positive convention
    angle_deg = -angle_deg
    
    return angle_deg

def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Smooth a binary mask using a kernel.
    Args:
        mask: binary mask to smooth
        kernel_size: size of the kernel
    Returns:
        Smoothed binary mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def get_mask_aabb(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get the axis-aligned bounding box of the mask.
    Args:
        mask: binary mask of the segmentation
    Returns:
        Tuple of (x, y, width, height) of the bounding box
    """

    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (0, 0, 0, 0)
        
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    return x, y, w, h