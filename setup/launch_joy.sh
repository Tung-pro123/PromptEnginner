#!/bin/bash
# Script khởi động tay cầm (Joystick) và phát dữ liệu lên ROS Topic (/joy)

echo "--- Khởi động ROS Joy Node ---"

# 1. Source ROS Workspace
# Sắp xếp source ROS từ hệ thống (/opt/ros/...) trước để đảm bảo có các lệnh cơ bản như rosrun
ROS_SETUP_BASH=$(ls /opt/ros/*/setup.bash 2>/dev/null | head -n 1)
if [ -n "$ROS_SETUP_BASH" ]; then
    source "$ROS_SETUP_BASH"
    echo "[OK] Sourced system ROS: $ROS_SETUP_BASH"
else
    echo "[WARNING] Không tìm thấy ROS cài đặt trên hệ thống (/opt/ros/*/setup.bash)"
fi

# Sắp xếp source thêm workspace cá nhân để có các package tự build
SETUP_BASH_PATH=$(find "$HOME" -maxdepth 3 -name "setup.bash" | grep "devel/setup.bash" | head -n 1)
if [ -z "$SETUP_BASH_PATH" ]; then
    SETUP_BASH_PATH="$HOME/catkin_ws/devel/setup.bash"
fi

if [ -f "$SETUP_BASH_PATH" ]; then
    source "$SETUP_BASH_PATH"
    echo "[OK] Sourced ROS workspace: $SETUP_BASH_PATH"
fi

# 2. Start roscore if not running
if ! pgrep -x "roscore" > /dev/null && ! pgrep -x "rosmaster" > /dev/null; then
    echo "Starting ROS Master (roscore)..."
    roscore &
    sleep 3
fi

# Kiểm tra xem tay cầm đã được kết nối với Jetson chưa
if [ ! -e /dev/input/js0 ]; then
    echo "[CẢNH BÁO] Không tìm thấy thiết bị joystick tại /dev/input/js0"
    echo "Hãy kiểm tra lại kết nối USB/Bluetooth của tay cầm!"
else
    echo "[INFO] Đã tìm thấy tay cầm tại /dev/input/js0"
    
    # Cấp quyền đọc/ghi cho cổng input để tránh lỗi Permission Denied
    sudo chmod a+rw /dev/input/js0
fi

echo "[INFO] Đang bắt đầu gửi tín hiệu lên topic /joy..."
echo "Nhấn Ctrl+C để dừng."

# Chạy thư viện joy của ROS, trỏ vào cổng js0 và thêm vùng đệm (deadzone) 5% chống trôi cần gạt
rosrun joy joy_node _dev:=/dev/input/js0 _deadzone:=0.05
