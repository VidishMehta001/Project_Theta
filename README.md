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
  
  
### For Vidish
  1. Put a folder called models in the workspace directory and put your models in the folder
  2. Build with "colcon build" ("--packeges-select theta_pkgs", if you already built everything else)
  3. Run the launchfile "ros2 launch src/launch/webcam_launch.py".
  Launch file needs to be run from the main workspace folder due to the location of the model files. (Need to adjust this in the future)

### To launch Nav2

  1. Launch with map: ros2 launch nav2_bringup bringup_launch.py use_sim_time:=False autostart:=False map:=/path/to/your-map.yaml then launch RVIZ ros2 run rviz2 rviz2 -d $(ros2 pkg prefix nav2_bringup)/share/nav2_bringup/launch/nav2_default_view.rviz
  2. Launch without map: ros2 launch nav2_bringup navigation_launch.py
