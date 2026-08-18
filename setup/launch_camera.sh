#!/bin/bash

# ==========================================
# LAUNCH CAMERA SCRIPT
# ==========================================

echo -e "\e[1;33m[1/3] Kiểm tra và cài đặt gói gscam, image-view, rqt-image-view (nếu chưa có)...\e[0m"
sudo apt-get install -y ros-melodic-gscam ros-melodic-image-view ros-melodic-rqt-image-view

# 1. Source ROS Workspace
SETUP_BASH="$HOME/catkin_ws/devel/setup.bash"
if [ -f "$SETUP_BASH" ]; then
    source "$SETUP_BASH"
    echo -e "\e[1;32m[OK] Sourced ROS workspace: $SETUP_BASH\e[0m"
else
    echo -e "\e[1;31m[ERROR] Could not find ROS setup.bash at $SETUP_BASH\e[0m"
    echo -e "Please build your catkin workspace first."
    exit 1
fi

# 2. Start roscore if not running
if ! pgrep -x "roscore" > /dev/null && ! pgrep -x "rosmaster" > /dev/null; then
    echo -e "\e[1;33mStarting ROS Master (roscore)...\e[0m"
    roscore &
    sleep 3
fi

# 3. Launch Camera
echo -e "\n\e[1;34m--- STARTING CAMERA ---\e[0m"
roslaunch jetracer csi_camera.launch &
CAMERA_PID=$!

echo -e "\n\e[1;32m=== CAMERA IS RUNNING ===\e[0m"
echo "Camera PID: $CAMERA_PID"
echo -e "\e[1;31mPress Ctrl+C to stop Camera.\e[0m"

# Handle graceful shutdown
trap "echo -e '\nStopping Camera...'; kill $CAMERA_PID 2>/dev/null; killall -9 nvargus_daemon_client jetson_camera 2>/dev/null; exit" SIGINT SIGTERM

wait
