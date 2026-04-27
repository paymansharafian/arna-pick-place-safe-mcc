#!/bin/bash
# ARNA Legion launch script
# Forces ROS to bind to the ARNA WiFi interface (10.0.0.101)
# even when a second ethernet interface (public internet) is present.

export ROS_IP=10.0.0.101
export ROS_MASTER_URI=http://10.0.0.101:11311
roslaunch pick_place pick_place.launch
