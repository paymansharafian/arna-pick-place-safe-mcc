#!/usr/bin/python3

import actionlib.goal_id_generator
from kortex_driver.msg import FollowCartesianTrajectoryAction, FollowCartesianTrajectoryActionGoal, FollowCartesianTrajectoryGoal, CartesianWaypoint, Pose as CartesianPose
from kortex_driver.msg import Gripper, Finger, GripperCommand
from kortex_driver.srv import SendGripperCommand
import rospy
import actionlib

from transform import *

# wait for service to be ready
rospy.wait_for_service('my_gen3/base/send_gripper_command')

trajectory_action = actionlib.SimpleActionClient('/my_gen3/cartesian_trajectory_controller/follow_cartesian_trajectory', FollowCartesianTrajectoryAction)
trajectory_action.wait_for_server()
grip_srv = rospy.ServiceProxy('my_gen3/base/send_gripper_command', SendGripperCommand)

print("Created services")

def arm_tool_position():
    return get_frame_position("tool_frame", "base_link")

def arm_tool_rotation():
    return get_frame_rotation_euler("tool_frame", "base_link")

def arm_set_pose(position: Point3D, orientation: Point3D):
    if (rospy.is_shutdown()):
        return
    
    print("Moving arm")
    
    goal = FollowCartesianTrajectoryGoal()
    goal.trajectory.append(CartesianWaypoint(CartesianPose(position.x, position.y, position.z, orientation.x, orientation.y, orientation.z), 0, 0.1, 20, 0))
    goal.use_optimal_blending = True

    trajectory_action.send_goal_and_wait(goal, rospy.Duration(10), rospy.Duration(10))
    print("Arm moved")
    rospy.sleep(0.1)

def arm_translate(offset: Point3D):
    position = arm_tool_position()
    arm_set_pose(Point3D(position.x + offset.x, position.y + offset.y, position.z + offset.z), get_frame_rotation_euler("tool_frame", "base_link"))

def arm_rotate(offset: Point3D):
    rotation = arm_tool_rotation()
    arm_set_pose(arm_tool_position(), Point3D(rotation.x + offset.x, rotation.y + offset.y, rotation.z + offset.z))

def arm_rotate_tool(offset: Point3D):
    local_offset = transform_pyrotation(offset, "tool_frame", "base_link")
    arm_set_rotation(local_offset)

def arm_translate_tool(offset: Point3D):
    local_offset = transform_pypoint(offset, "tool_frame", "base_link")
    arm_set_position(local_offset)

def arm_set_position(position: Point3D):
    arm_set_pose(position, arm_tool_rotation())

def arm_set_rotation(rotation: Point3D):
    arm_set_pose(arm_tool_position(), rotation)

def grip(amount):
    grip_srv(GripperCommand(3, Gripper([Finger(0, amount)]), 0))
