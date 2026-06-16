#!/usr/bin/python3

import actionlib.goal_id_generator
from kortex_driver.msg import FollowCartesianTrajectoryAction, FollowCartesianTrajectoryActionGoal, FollowCartesianTrajectoryGoal, CartesianWaypoint, Pose as CartesianPose
from kortex_driver.msg import Gripper, Finger, GripperCommand
from kortex_driver.srv import SendGripperCommand
from kortex_driver.msg import Action, ActionHandle, Action_action_parameters, ConstrainedJointAngles, JointAngles, JointAngle
from kortex_driver.msg import ConstrainedPose, CartesianSpeed
from kortex_driver.srv import ExecuteAction, ExecuteActionRequest
from kortex_driver.srv import GetMeasuredCartesianPose
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
get_measured_pose_srv = rospy.ServiceProxy('my_gen3/base/get_measured_cartesian_pose', GetMeasuredCartesianPose)
get_measured_pose_srv.wait_for_service()

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

def _get_measured_pose():
    """Current Cartesian pose from the arm in Kinova-native convention
    (x,y,z metres, theta_x/y/z degrees), or None on failure."""
    try:
        return get_measured_pose_srv().output
    except Exception as e:
        rospy.logerr("[arm_cmd] get_measured_cartesian_pose failed: %s" % e)
        return None

# Cartesian translation speed cap for REACH_POSE moves (m/s).  Matches the
# 0.1 m/s the old streaming waypoint used — safe for table-top approach.
REACH_SPEED_M_S = 0.10

def arm_set_position(position: Point3D, timeout: float = 15.0, tol: float = 0.012):
    """Move the tool to `position` (base_link, metres), keeping the current
    orientation, using the Kinova ONBOARD motion generator (REACH_POSE).

    Why not the cartesian_trajectory_controller: that controller streams
    low-level setpoints and the arm's servo watchdog aborts the trajectory
    (sub-error 140) when commanded from a near-singular top-down viewing pose
    or under remote-network latency.  REACH_POSE runs entirely on the arm and
    is immune to both — the same subsystem arm_home() uses successfully.

    The goal is a PURE TRANSLATION applied to the arm's own measured pose:
    target = measured_pose + (position − current_tool_position), orientation
    unchanged.  Because it is a rigid-body translation with no rotation, the
    result is independent of which tool reference the firmware uses and of the
    Euler-angle convention — both cancel out.  Orientation is round-tripped
    from the measured pose verbatim, so no radians/degrees conversion is done.

    Returns True if the tool reaches `position` within `tol` metres before
    `timeout`, else False (treat as a failed stage, like the old contract).
    """
    if rospy.is_shutdown():
        return False

    measured = _get_measured_pose()
    tf_now   = get_frame_position("tool_frame", "base_link")
    if measured is None or tf_now is None:
        rospy.logerr("[arm_cmd] reach_pose: could not read current pose — aborting move")
        print("Arm move FAILED")
        return False

    dx = float(position.x) - float(tf_now.x)
    dy = float(position.y) - float(tf_now.y)
    dz = float(position.z) - float(tf_now.z)

    target = CartesianPose(
        measured.x + dx, measured.y + dy, measured.z + dz,
        measured.theta_x, measured.theta_y, measured.theta_z,
    )

    constrained = ConstrainedPose()
    constrained.target_pose = target
    constrained.constraint.oneof_type.speed.append(
        CartesianSpeed(REACH_SPEED_M_S, 30.0))

    action = Action()
    action.name = "pick_reach_pose"
    action.application_data = ""
    action.handle = ActionHandle()
    action.handle.identifier = 1001
    action.handle.action_type = 6   # REACH_POSE
    action.handle.permission = 1
    action.oneof_action_parameters = Action_action_parameters()
    action.oneof_action_parameters.reach_pose.append(constrained)

    req = ExecuteActionRequest()
    req.input = action
    print("Moving arm (reach_pose)")
    execute_action(req)

    # Wait for convergence by polling the tool frame in tf (same frame the
    # target is expressed in, so convergence is meaningful regardless of the
    # firmware's internal tool reference).  Bail early if the arm stalls.
    deadline = rospy.Time.now() + rospy.Duration(timeout)
    rate     = rospy.Rate(20)
    best_d   = None
    best_t   = rospy.Time.now()
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        cur = get_frame_position("tool_frame", "base_link")
        if cur is not None:
            d = ((float(cur.x) - float(position.x)) ** 2
                 + (float(cur.y) - float(position.y)) ** 2
                 + (float(cur.z) - float(position.z)) ** 2) ** 0.5
            if d <= tol:
                print("Arm moved (reach_pose)")
                rospy.sleep(0.1)
                return True
            if best_d is None or d < best_d - 0.002:
                best_d, best_t = d, rospy.Time.now()
            elif best_d > 0.03 and (rospy.Time.now() - best_t).to_sec() > 3.0:
                rospy.logerr("[arm_cmd] reach_pose: stalled %.3fm from target — aborting move" % best_d)
                print("Arm move FAILED")
                return False
        rate.sleep()

    rospy.logerr("[arm_cmd] reach_pose: did not converge within %.1fs (best dist=%s)"
                 % (timeout, best_d))
    print("Arm move FAILED")
    return False

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