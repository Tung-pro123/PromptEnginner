#!/bin/bash

# ==========================================
# LAUNCH LIDAR SCRIPT
# ==========================================

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

# 3. Launch LiDAR
echo -e "\n\e[1;34m--- STARTING LIDAR (Topic: /scan) ---\e[0m"
# Ép (remap) tất cả đầu ra của lidar.launch về đúng topic /scan
roslaunch jetracer lidar.launch scan_topic:=/scan /scan:=/scan &
LIDAR_PID=$!

echo -e "\n\e[1;32m=== LIDAR IS RUNNING ===\e[0m"
echo "LiDAR PID: $LIDAR_PID"
echo -e "\e[1;31mPress Ctrl+C to stop LiDAR.\e[0m"

# Handle graceful shutdown
trap "echo -e '\nStopping LiDAR...'; kill $LIDAR_PID 2>/dev/null; killall -9 ydlidar_node rplidarNode 2>/dev/null; exit" SIGINT SIGTERM

wait
