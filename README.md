# Project Theta
Project Theta is a assistant for the visually impared using SLAM, navigation and object detection to guide them to household objects. 

## Instructions
  1. Follow the instructions here to install ROS2 Foxy https://docs.ros.org/en/foxy/index.html
  2. Create a workspace (ros2_ws) in your home directory
  3. Create a folder called src in your workspace (ros2_ws/src)
  4. Clone the project into your src folder
  5. Ensure that all requirements for submodules are completed (RTABMAP_ROS, IMAGE_PIPELINE, DEPTHAI)
  6. Go to your ros2_ws and run "source /opt/ros/foxy/setup.bash" in the terminal
  7. Run "colcon build" in the terminal
  8. Run "source install/setup.bash" in the terminal
  9. Run the launch file "ros2 launch src/launch/theta_launch.py
  10. In another terminal run "source install/setup.bash"
  11. Run the odometery node "ros2 run rtabmap_ros stereo_odometry --ros-args -p approx_sync:=true" 
  (This has to be done seperately as odometery fails when running in the launch file for some reason).

