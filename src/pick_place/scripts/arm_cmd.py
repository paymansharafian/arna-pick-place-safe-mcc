#!/usr/bin/python3

import actionlib.goal_id_generator
from kortex_driver.msg import FollowCartesianTrajectoryAction, FollowCartesianTrajectoryActionGoal, FollowCartesianTrajectoryGoal, CartesianWaypoint, Pose as CartesianPose
from kortex_driver.msg import Gripper, Finger, GripperCommand
from kortex_driver.srv import SendGripperCommand
from kortex_driver.msg import Action, ActionHandle, Action_action_parameters, ConstrainedJointAngles, JointAngles, JointAngle
from kortex_driver.srv import ExecuteAction, ExecuteActionRequest
from kortex_driver.srv import StopAction

import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus

from transform import *

# wait for service to be ready
execute_action = rospy.ServiceProxy('my_gen3/base/execute_action', ExecuteAction)
execute_action.wait_for_service()
trajectory_action = actionlib.SimpleActionClient('/my_gen3/cartesian_trajectory_controller/follow_cartesian_trajectory', FollowCartesianTrajectoryAction)
trajectory_action.wait_for_server()
grip_srv = rospy.ServiceProxy('my_gen3/base/send_gripper_command', SendGripperCommand)
grip_srv.wait_for_service()
stop_action_srv = rospy.ServiceProxy('my_gen3/base/stop_action', StopAction)
stop_action_srv.wait_for_service()

print("Created services")

def arm_tool_position():
    return get_frame_position("tool_frame", "base_link")

def arm_tool_rotation():
    return get_frame_rotation_euler("tool_frame", "base_link")

def arm_set_pose(position: Point3D, orientation: Point3D):
    """Send a single-waypoint Cartesian trajectory and WAIT for it.

    Returns True only if the arm controller reported the goal SUCCEEDED.
    A returned False means the trajectory was aborted (e.g. Kinova sub-error
    140, CONTROL_WAYPOINT_TRAJECTORY_ABORTED) or rejected (e.g. waypoint
    outside the robot workspace). Callers MUST treat False as a failed stage
    and abort — never silently continue from a wrong pose.
    """
    if (rospy.is_shutdown()):
        return False

    print("Moving arm")

    goal = FollowCartesianTrajectoryGoal()
    goal.trajectory.append(CartesianWaypoint(CartesianPose(position.x, position.y, position.z, orientation.x, orientation.y, orientation.z), 0, 0.1, 20, 0))
    goal.use_optimal_blending = True

    state = trajectory_action.send_goal_and_wait(goal, rospy.Duration(10), rospy.Duration(10))
    result = trajectory_action.get_result()

    ok = (state == GoalStatus.SUCCEEDED)
    if ok and result is not None and hasattr(result, 'error_code'):
        ok = (result.error_code == 0)

    if ok:
        print("Arm moved")
    else:
        err = getattr(result, 'error_code', None)
        errstr = getattr(result, 'error_string', '')
        rospy.logerr("[arm_cmd] Arm move FAILED: state=%s error_code=%s %s"
                     % (state, err, errstr))
        print("Arm move FAILED")

    rospy.sleep(0.1)
    return ok

def arm_translate(offset: Point3D):
    position = arm_tool_position()
    return arm_set_pose(Point3D(position.x + offset.x, position.y + offset.y, position.z + offset.z), get_frame_rotation_euler("tool_frame", "base_link"))

def arm_rotate(offset: Point3D):
    rotation = arm_tool_rotation()
    return arm_set_pose(arm_tool_position(), Point3D(rotation.x + offset.x, rotation.y + offset.y, rotation.z + offset.z))

def arm_rotate_tool(offset: Point3D):
    local_offset = transform_pyrotation(offset, "tool_frame", "base_link")
    return arm_set_rotation(local_offset)

def arm_translate_tool(offset: Point3D):
    local_offset = transform_pypoint(offset, "tool_frame", "base_link")
    return arm_set_position(local_offset)

def arm_set_position(position: Point3D):
    return arm_set_pose(position, arm_tool_rotation())

def arm_set_rotation(rotation: Point3D):
    return arm_set_pose(arm_tool_position(), rotation)

def grip(amount):
    grip_srv(GripperCommand(3, Gripper([Finger(0, amount)]), 0))


def arm_home():
    # Create the action message
    action = Action()
    
    # Set up the action handle
    action.handle = ActionHandle()
    action.handle.identifier = 2
    action.handle.action_type = 7
    action.handle.permission = 1
    
    # Set the action name and application data
    action.name = "Home"
    action.application_data = ""
    
    # Create the joint angles message
    joint_angles = JointAngles()
    angles = [
        (0, 0.0),
        (1, 15.0),
        (2, 180.0),
        (3, 230.0),
        (4, 0.0),
        (5, 55.0),
        (6, 90.0)
    ]
    
    # Add each joint angle to the message
    for joint_id, value in angles:
        joint_angle = JointAngle()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = value
        joint_angles.joint_angles.append(joint_angle)
    
    # Create the constrained joint angles message
    constrained_joint_angles = ConstrainedJointAngles()
    constrained_joint_angles.joint_angles = joint_angles
    constrained_joint_angles.constraint.type = 0
    constrained_joint_angles.constraint.value = 0.0
    
    # Set up the action parameters
    action.oneof_action_parameters = Action_action_parameters()
    action.oneof_action_parameters.reach_joint_angles.append(constrained_joint_angles)
    
    # Create and send the service request
    req = ExecuteActionRequest()
    req.input = action
    
    execute_action(req)