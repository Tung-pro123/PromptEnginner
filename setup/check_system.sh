#!/bin/bash

# ==========================================
# SYSTEM CHECK SCRIPT
# ==========================================

echo -e "\e[1;34m=== SYSTEM CHECK ===\e[0m"

echo -e "\n\e[1;33m1. Checking Disk Space:\e[0m"
df -h /

echo -e "\n\e[1;33m2. Checking Memory:\e[0m"
free -m

echo -e "\n\e[1;33m3. Checking USB Devices (LiDAR):\e[0m"
ls -l /dev/ttyUSB* 2>/dev/null || echo -e "\e[0;31mNo /dev/ttyUSB* devices found.\e[0m"
ls -l /dev/ydlidar 2>/dev/null || echo -e "\e[0;31mNo /dev/ydlidar symlink found.\e[0m"

echo -e "\n\e[1;33m4. Checking Video Devices (Camera):\e[0m"
ls -l /dev/video* 2>/dev/null || echo -e "\e[0;31mNo /dev/video* devices found.\e[0m"

echo -e "\n\e[1;33m5. Checking Jetson Temperature:\e[0m"
if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
    temp=$(cat /sys/class/thermal/thermal_zone0/temp)
    echo "Temperature: $((temp/1000)) °C"
else
    echo "Temperature sensor not found."
fi

echo -e "\n\e[1;32m=== CHECK COMPLETE ===\e[0m"
