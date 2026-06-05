#!/usr/bin/env python3
"""
arm_cmd_passthrough_node — Phase 1 bypass relay (enable_mpc_cbf:=false).

Subscribes to /my_gen3/in/cartesian_velocity_desired and immediately
republishes every message verbatim on /my_gen3/in/cartesian_velocity with
zero processing.  Exactly one of this node or mpc_cbf_arm_node is running
at any time, ensuring no commands are silently dropped.
"""

import rospy
from kortex_driver.msg import TwistCommand
from std_msgs.msg import Bool


def main():
    rospy.init_node('arm_cmd_passthrough_node', anonymous=False)

    pub = rospy.Publisher(
        '/my_gen3/in/cartesian_velocity', TwistCommand, queue_size=1)

    pick_running = False

    def pick_cb(msg):
        nonlocal pick_running
        pick_running = msg.data

    def cb(msg):
        if pick_running:
            return   # suppress during ExecuteAction trajectories (homing/pick)
        pub.publish(msg)

    rospy.Subscriber('/pick_running', Bool, pick_cb, queue_size=1)
    rospy.Subscriber(
        '/my_gen3/in/cartesian_velocity_desired', TwistCommand, cb,
        queue_size=1)

    rospy.loginfo(
        '[arm_cmd_passthrough] Ready — relaying '
        'cartesian_velocity_desired → cartesian_velocity')
    rospy.spin()


if __name__ == '__main__':
    main()
