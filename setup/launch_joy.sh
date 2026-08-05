#!/bin/bash
# Script khởi động tay cầm (Joystick) và phát dữ liệu lên ROS Topic (/joy)

echo "--- Khởi động ROS Joy Node ---"

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
