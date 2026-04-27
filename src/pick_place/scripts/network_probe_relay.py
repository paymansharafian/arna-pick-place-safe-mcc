#!/usr/bin/env python3
"""
network_probe_relay — Phase 0 ping/pong relay node.

Subscribes to /network_probe_ping and immediately republishes each message
verbatim on /network_probe_pong.  Runs on Legion so that the round-trip
for a browser-originated ping traverses:

  browser → Cloudflare → rosbridge (port 9090) → ROS graph
                                  ↑ this relay ↓
  browser ← Cloudflare ← rosbridge (port 9090) ← ROS graph

The RTT measured by the browser therefore captures the full Cloudflare +
rosbridge WebSocket path in both directions.
"""

import rospy
from std_msgs.msg import String

def main():
    rospy.init_node('network_probe_relay', anonymous=False)

    pub = rospy.Publisher('/network_probe_pong', String, queue_size=10)

    def ping_cb(msg):
        pub.publish(msg)

    rospy.Subscriber('/network_probe_ping', String, ping_cb, queue_size=10)
    rospy.loginfo('[network_probe_relay] Ready — relaying /network_probe_ping → /network_probe_pong')
    rospy.spin()

if __name__ == '__main__':
    main()
