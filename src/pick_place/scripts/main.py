#!/usr/bin/env python3
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

# output publishers
image_pub = rospy.Publisher('pick_place_cam', Image, queue_size=1)
image_compressed_pub = rospy.Publisher('pick_place_cam/compressed', CompressedImage, queue_size=1)
pick_ready_pub = rospy.Publisher('pick_ready', Bool, queue_size=1)

window_name = 'Pick Place'

cam = CameraStream(
    color_topic='/camera/color/image_raw',
    depth_topic='/camera/depth_registered/sw_registered/image_rect_raw',
    color_info_topic='/camera/color/camera_info',
    depth_info_topic='/camera/depth/camera_info'
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

    starting_tool_position = get_frame_position("tool_frame", "base_link")

    target = get_mask_point(mask, depth, color_info)
    target = transform_pypoint(target, "camera_color_frame", "base_link")

    # vector along tool plane
    pointing_normal = get_frame_axis("tool_frame", "base_link", axis='z')
    tool_position = get_frame_position("tool_frame", "base_link")

    # tool plane
    tool_plane = Plane(tool_position, normal_vector=pointing_normal)

    # get camera position and project onto tool plane
    cam_position = get_frame_position("camera_color_frame", "base_link")
    camera_projected = tool_plane.projection(cam_position)

    # project target position onto tool plane
    target_projected = tool_plane.projection(target)

    # add this vector to tool frame
    aligned_position = tool_position + (target_projected - camera_projected)

    # translate arm by this vector
    arm_set_position(aligned_position)
    print("aligned camera")

    # get the plane at the target
    target_plane = Plane(target, normal_vector=pointing_normal)

    # project the tool position onto the target plane so we can approach while keeping the object centered in the frame
    tool_position = get_frame_position("tool_frame", "base_link")
    tool_projected = target_plane.projection(tool_position)

    # get the max percentage of the frame w or h that the segmented object bb takes up
    x,y,w,h = get_mask_aabb(mask)
    max_percent = max(w / 640, h / 480)

    # get position that percentage of the way towards the projected tool position
    intermediate_target = lerp_point(tool_position, tool_projected, (1 - max_percent))

    # move until the object fills the frame
    arm_set_position(intermediate_target)
    print("filled frame with object")

    # resegment at center of image once more
    mask = segment_image(get_frame_center(), color)
    target = get_mask_point(mask, depth, color_info)
    target = transform_pypoint(target, "camera_color_frame", "base_link")

    # Get the detected angle from segmentation
    angle_rad = math.radians(get_mask_rotation(mask))

    # align gripper to the target now (rather than camera)
    tool_position = get_frame_position("tool_frame", "base_link")
    tool_plane = Plane(tool_position, normal_vector=pointing_normal)
    target_projected = tool_plane.projection(target)

    arm_set_position(target_projected)
    print("final aligned")

    mask = None
    mouse_click_point = None

    # rotate
    arm_rotate_tool(Point3D(0, 0, angle_rad))

    # Open gripper
    grip(0)

    # move to final position
    arm_set_position(target)
    # Close gripper
    grip(1)

    # move back to starting tool position
    arm_set_position(starting_tool_position)

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