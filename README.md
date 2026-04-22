The ARNA pick\_place package allows the Kinova arm to autonomously pick up objects and operate the arm through a GUI. 
It uses a system external to ARNA to run the package in its current state due to dependency issues with ARNA’s outdated firmware, 
however it should be integrated into ARNA’s ROS system in the future after ARNA has been upgraded (preferably to ROS 2 or ROS noetic).

System:
- Ubuntu 20.04  
- ROS Noetic

Dependencies

- ROS Dependencies  
  - Kortex Driver  
    - [https://github.com/Kinovarobotics/ros\_kortex](https://github.com/Kinovarobotics/ros_kortex)   
  - Kortex Vision  
    - [https://github.com/Kinovarobotics/ros\_kortex\_vision](https://github.com/Kinovarobotics/ros_kortex_vision)   
  - rosbridge\_server  
  - web\_video\_server  
  - Use rosdep to install other dependencies  
- Python 3.8+  
  - All Python dependencies are in the requirements.txt  
- Catkin build

Compilation

- Install ROS Noetic & make sure Python 3.8+ is available  
- Clone Repository from GitHub: https://github.com/LARRILabs/ARNA_PICK_PLACE/
- Change Directories into ARNA\_PICK\_PLACE/src/  
- Clone and follow install directions from Kortex Driver and Vision GitHubs  
- Run the following commands  
  - sudo apt-get install ros-noetic-rosbridge-server ros-noetic-web-video-server  
  - sudo apt-get install python3-rosdep  
  - rosdep install \--from-paths src \--ignore-src \-r \-y  
  - sudo apt-get install python3-catkin-tools  
- Change Directories into ARNA\_PICK\_PLACE/src/pick\_place  
- Run the following commands  
  - pip install \-r requirements.txt

Execution and User Experience

- Turn on ARNA and the kinova arm  
- Connect to the ARNA wifi  
- In a terminal run; “roslaunch pick\_place pick\_place.launch”  
- NATHAN EXPLAIN GUI

The pick\_place code consists of 2 main functions, segment image and execute movement.

Segment image function

- When the image is clicked, it uses that point and runs FastSAM  
  - FastSAM is the Ultralytics YOLO, segment anything model, fast version.  
  - The model estimates and segments the object that was clicked on  
- The segmented image is turned into a mask, which is applied to both the color and depth images using a bitwise and  
- The rotation of the object is found (uses this angle for visualization, not grasping yet)  
  - The mask is made into a contour, then a minimum-area bounding rectangle is fitted around it.  
  - The change in x and y across one of the longer sides are used to get the arc-tangent of the 2, atan(dx/dy), then converted to degrees.  
  - Degrees are normalized to be from \-90 to 90, instead of 0 to 180\.  
- Grabs the center of the mask  
  - uses the moments to approximate the center of mass  
- Gets the closest point  
  - x and y come from the center of the mask  
  - Valid depths all non zero depth values inside the mask  
  - A final Z value is taken by first getting the average of all valid depths, then taking the average of all depths which are greater than the first average.  
  - X, Y, and Z are converted to position coordinates from pixel coordinates.  
    - X pos \= (\[ X pix – center X \] \* Z) / focal length X  
    - Y pos \= (\[ Y pix – center Y \] \* Z) / focal length Y

Execute movement

- Grabs the starting tool position, axis in which the tool is pointing and camera position  
- Projects the camera position onto the tool plane, defined by the tool position and axis its pointing in  
- Project the target position onto the tool frame and align the frames together with the tool position  
- Physically align the camera to the object, getting it in the center of the camera to improve detection and grasping  
- Resegments the image, using the center of the camera as the segmentation point  
- Approach the object, keeping it in the center by moving along the normal of the object  
- Move forward until the object takes up the majority of the image  
- Resegment  
- Get angle of the object via method discussed above  
- Align the gripper (no longer camera) to the object, then rotate to match the angle  
- Open gripper, move to final position, close gripper, move back to starting position

User perspective

* Window shows the camera feed from the kinova arm  
* Click anywhere on the image and it will change it to a segmented version to show what object is chosen.  
  * Runs segmentation function  
  * Uses FastSAM to segment based on the clicked point  
  * Makes the object into a mask on the depth and color images  
  * Uses the estimated center of mass for the pixel X and Y  
  * Uses the average of the above average values on the depth as the Z  
  * Transforms the pixel X and Y to real X and Y via the depth and camera info  
* If you don’t like the object you can reset the image and try again.  
* If you do, you can tell the arm to pick up the object.   
  * Runs Execute Movement on the point chosen in segmentation  
  * Align the camera with the object, resegment  
  * Move towards the object so it fills most of the image, resegment  
  * Move to final location, with the point translated to gripper frame from camera frame  
  * Grab the object  
* Once it grabs the object it resets to the default position

TODO

Add to instructions

- ARNA.network while connected to ARNA router

Add in pics and videos of it working
