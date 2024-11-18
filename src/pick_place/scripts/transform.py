import math
import sys
from typing import Optional, Union, Tuple

import tf2_geometry_msgs
import tf2_ros
import tf
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, TransformStamped, Quaternion
import rospy
from sympy import Point3D, Matrix

# Create tf buffer and listener globally - they are thread-safe
_tf_buffer = tf2_ros.Buffer()
_tf_listener = tf2_ros.TransformListener(_tf_buffer)

def to_sympy_point(point: Union[Point, PointStamped]) -> Point3D:
    """Convert ROS Point/PointStamped to sympy Point3D."""
    if isinstance(point, PointStamped):
        point = point.point
    return Point3D(point.x, point.y, point.z)

def to_sympy_vector(point: Union[Point, PointStamped]) -> Point3D:
    """Convert ROS Point/PointStamped to sympy Point3D for vector representation."""
    if isinstance(point, PointStamped):
        point = point.point
    return Point3D(point.x, point.y, point.z)

def to_ros_point(point: Union[Point3D, Tuple[float, float, float]]) -> Point:
    """Convert sympy Point3D to ROS Point."""
    if isinstance(point, Point3D):
        return Point(point.x, point.y, point.z)
    return Point(point[0], point[1], point[2])

def to_ros_point_stamped(point: Point3D, frame_id: str) -> PointStamped:
    """Convert sympy Point3D to ROS PointStamped."""
    point_stamped = PointStamped()
    point_stamped.point = to_ros_point(point)
    point_stamped.header.frame_id = frame_id
    point_stamped.header.stamp = rospy.Time.now()
    return point_stamped

def quaternion_to_rotation_matrix(q: Quaternion) -> Matrix:
    """Convert ROS Quaternion to sympy rotation Matrix."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return Matrix([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def get_rotation_quaternion(angle_deg: float) -> Quaternion:
    """Convert angle in degrees to a quaternion representing rotation around z-axis."""
    angle_rad = math.radians(angle_deg)
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle_rad))

def transform_frames(
    source_frame: str,
    target_frame: str,
    timeout: float = 1.0
) -> Optional[TransformStamped]:
    """Get the transform from source_frame to target_frame using tf2."""
    try:
        transform = _tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(timeout)
        )
        return transform
        
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to get transform: {str(e)}")
        return None

def transform_pose(
    pose: Union[Pose, PoseStamped],
    source_frame: str,
    target_frame: str,
    timeout: float = 1.0
) -> Optional[PoseStamped]:
    """Transform a pose from source_frame to target_frame."""
    try:
        if isinstance(pose, Pose):
            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            pose_stamped.header.frame_id = source_frame
            pose_stamped.header.stamp = rospy.Time.now()
        else:
            pose_stamped = pose
            
        transformed_pose = _tf_buffer.transform(
            pose_stamped,
            target_frame,
            rospy.Duration(timeout)
        )
        return transformed_pose
        
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to transform pose: {str(e)}")
        return None

def transform_point(
    point: Union[Point, PointStamped],
    source_frame: str,
    target_frame: str,
    timeout: float = 1.0
) -> Optional[Point3D]:
    """Transform a point from source_frame to target_frame."""
    try:
        # Create PointStamped if necessary
        if isinstance(point, Point):
            point_stamped = PointStamped()
            point_stamped.point = point
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = rospy.Time.now()
        else:
            point_stamped = point
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = rospy.Time.now()

        # Get transform
        transform = _tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(timeout)
        )

        # Transform the point
        transformed_point = tf2_geometry_msgs.do_transform_point(
            point_stamped,
            transform
        )
        
        return to_sympy_point(transformed_point)
        
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to transform point: {str(e)}")
        return None
    
def transform_pypoint(
    point: Union[Point3D, Tuple[float, float, float]],
    source_frame: str,
    target_frame: str,
    timeout: float = 1.0) -> Optional[Point3D]:
    """Transform a point from source_frame to target_frame."""
    try:
        # Create PointStamped
        point_stamped = PointStamped()
        point_stamped.point = to_ros_point(point)
        point_stamped.header.frame_id = source_frame
        point_stamped.header.stamp = rospy.Time.now()

        # Get transform
        transform = _tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(timeout)
        )

        # Transform the point
        transformed_point = tf2_geometry_msgs.do_transform_point(
            point_stamped,
            transform
        )
        
        return to_sympy_point(transformed_point)
        
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to transform point: {str(e)}")
        return None

def transform_pyrotation(rotation: Point3D, source_frame: str, target_frame: str, timeout: float = 1.0) -> Optional[Point3D]:
    """Transform a rotation from source_frame to target_frame."""
    try:
        # Create Quaternion
        q = tf.transformations.quaternion_from_euler(rotation.x, rotation.y, rotation.z)
        quaternion = Quaternion(*q)
        
        # Create PoseStamped
        pose_stamped = PoseStamped()
        pose_stamped.pose.orientation = quaternion
        pose_stamped.header.frame_id = source_frame
        pose_stamped.header.stamp = rospy.Time.now()

        # Get transform
        transform = _tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(timeout)
        )

        # Transform the pose
        transformed_pose = tf2_geometry_msgs.do_transform_pose(
            pose_stamped,
            transform
        )
        
        # Convert quaternion to Euler angles
        transformed_quaternion = transformed_pose.pose.orientation
        q = (transformed_quaternion.x, transformed_quaternion.y, transformed_quaternion.z, transformed_quaternion.w)
        euler = tf.transformations.euler_from_quaternion(q)
        return Point3D(*euler)
        
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to transform rotation: {str(e)}")
        return None

def get_frame_axis(
    source_frame: str,
    target_frame: str,
    axis: str = 'z',
    timeout: float = 1.0
) -> Optional[Point3D]:
    """Get a vector representing the specified axis of source_frame in target_frame's space."""
    try:
        # Create axis point and origin in source frame
        axis_point = Point()
        if axis.lower() == 'x':
            axis_point.x = 1.0
        elif axis.lower() == 'y':
            axis_point.y = 1.0
        else:  # z axis by default
            axis_point.z = 1.0
            
        origin_point = Point()
        
        # Transform both points
        transformed_axis = transform_point(axis_point, source_frame, target_frame, timeout)
        transformed_origin = transform_point(origin_point, source_frame, target_frame, timeout)
        
        if transformed_axis is None or transformed_origin is None:
            return None
            
        # Calculate direction vector and normalize
        direction = transformed_axis - transformed_origin
        magnitude = float(direction.distance(Point3D(0, 0, 0)))
        if magnitude > 0:
            return direction / magnitude
        return direction
        
    except Exception as e:
        rospy.logerr(f"Failed to get frame axis: {str(e)}")
        return None

def get_frame_position(
    source_frame: str,
    target_frame: str,
    timeout: float = 1.0
) -> Optional[Point3D]:
    """Get the position of source_frame's origin in target_frame's space."""
    try:
        # Transform origin point
        origin = Point()
        transformed_origin = transform_point(origin, source_frame, target_frame, timeout)
        return transformed_origin
        
    except Exception as e:
        rospy.logerr(f"Failed to get frame position: {str(e)}")
        return None

def get_frame_rotation(
    source_frame: str,
    target_frame: str,
    timeout: float = 1.0
) -> Optional[Matrix]:
    """Get the rotation matrix of source_frame in target_frame's space."""
    try:
        # Get transform
        transform = _tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(timeout)
        )
        
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(transform.transform.rotation)
        return rotation_matrix
        
    except Exception as e:
        rospy.logerr(f"Failed to get frame rotation: {str(e)}")
        return None

def get_frame_rotation_euler(
    source_frame: str,
    target_frame: str,
    timeout: float = 1.0
) -> Optional[Point3D]:
    """Get the rotation of source_frame in target_frame's space as Euler angles."""
    try:
        # Get transform
        transform = _tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(timeout)
        )
        
        # Convert quaternion to Euler angles
        euler = tf.transformations.euler_from_quaternion([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ])
        return Point3D(*euler)
        
    except Exception as e:
        rospy.logerr(f"Failed to get frame rotation: {str(e)}")
        return None

def lerp_point(
    start: Point3D,
    end: Point3D,
    t: float
) -> Point3D:
    """Linearly interpolate between two points."""
    return start + t * (end - start)

def tuple_point_to_msg(point: Tuple[float, float, float], frame_id: str) -> PointStamped:
    """Convert tuple (x, y, z) to ROS PointStamped."""
    point_msg = PointStamped()
    point_msg.point.x = point[0]
    point_msg.point.y = point[1]
    point_msg.point.z = point[2]
    point_msg.header.frame_id = frame_id
    point_msg.header.stamp = rospy.Time.now()
    return point_msg