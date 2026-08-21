#!/bin/bash

# ==========================================
# CAMERA CALIBRATION SCRIPT (WAVESHARE)
# ==========================================

echo -e "\e[1;33m[1/4] Đang tải file Camera_overrides.tar.gz từ Waveshare...\e[0m"
wget https://files.waveshare.com/upload/e/eb/Camera_overrides.tar.gz

echo -e "\e[1;33m[2/4] Đang giải nén file...\e[0m"
tar zxvf Camera_overrides.tar.gz

echo -e "\e[1;33m[3/4] Copy file camera_overrides.isp vào thư mục hệ thống /var/nvidia/nvcam/settings/...\e[0m"
sudo mkdir -p /var/nvidia/nvcam/settings/
sudo cp camera_overrides.isp /var/nvidia/nvcam/settings/

echo -e "\e[1;33m[4/4] Cấp quyền và phân quyền sở hữu cho file...\e[0m"
sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp

echo -e "\e[1;32m[OK] Đã hoàn tất cài đặt camera_overrides.isp!\e[0m"

# Dọn dẹp file rác
echo -e "\e[1;33mĐang dọn dẹp file tải về...\e[0m"
rm Camera_overrides.tar.gz camera_overrides.isp 2>/dev/null
echo -e "\e[1;32mXong!\e[0m"
