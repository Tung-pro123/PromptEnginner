#!/bin/bash

# ==============================================================================
# Jetson AI Racer Challenge 2026 - Hardware & ROS Setup Script
# Board: NVIDIA Jetson Nano | Robot: Waveshare JetRacer Pro
# ==============================================================================

# Tự động tìm thư mục ROS Workspace chứa devel/setup.bash
SETUP_BASH_PATH=$(find "$HOME" -maxdepth 3 -name "setup.bash" | grep "devel/setup.bash" | head -n 1)
if [ -n "$SETUP_BASH_PATH" ]; then
    CATKIN_WS_DIR=$(dirname "$(dirname "$SETUP_BASH_PATH")")
else
    CATKIN_WS_DIR="$HOME/catkin_ws"
fi

# Tự động quét và phát hiện thư mục cài đặt ROS trên xe
ROS_SETUP_BASH=$(ls /opt/ros/*/setup.bash 2>/dev/null | head -n 1)
if [ -n "$ROS_SETUP_BASH" ]; then
    ROS_MELODIC_SETUP="$ROS_SETUP_BASH"
else
    ROS_MELODIC_SETUP="/opt/ros/melodic/setup.bash"
fi


# Màu sắc hiển thị
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}======================================================${NC}"
echo -e "${BLUE}${BOLD}    SETUP PHẦN CỨNG & CẢM BIẾN LIDAR - PROMPTENGINEER  ${NC}"
echo -e "${BLUE}${BOLD}======================================================${NC}"

# Kiểm tra nếu ROS chưa được cài đặt trên xe
if [ ! -f "$ROS_MELODIC_SETUP" ]; then
    echo -e "${RED}[ERROR] Không phát hiện bất kỳ bản cài đặt ROS nào trên hệ thống!${NC}"
    echo -e "Hệ thống cần có ROS để chạy các tính năng LiDAR và node điều khiển phần cứng."
    read -p "Bạn có muốn cài đặt tự động ROS Melodic (Dành cho Ubuntu 18.04 trên Jetson Nano) không? (y/n): " INSTALL_ROS
    if [[ "$INSTALL_ROS" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Bắt đầu cài đặt ROS Melodic...${NC}"
        
        echo -e "${YELLOW}1. Cấu hình apt sources.list cho ROS...${NC}"
        sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
        
        echo -e "${YELLOW}2. Thiết lập khóa bảo mật apt-key...${NC}"
        sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED0E1
        
        echo -e "${YELLOW}3. Cập nhật danh sách gói hệ thống...${NC}"
        sudo apt-get update
        
        echo -e "${YELLOW}4. Cài đặt ROS Melodic Base và các gói build liên quan...${NC}"
        sudo apt-get install -y ros-melodic-ros-base python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
        
        if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then
            echo -e "${YELLOW}5. Khởi tạo cơ sở dữ liệu rosdep...${NC}"
            sudo rosdep init
        fi
        rosdep update
        
        ROS_MELODIC_SETUP="/opt/ros/melodic/setup.bash"
        if [ -f "$ROS_MELODIC_SETUP" ]; then
            echo -e "${GREEN}[SUCCESS] Đã cài đặt ROS Melodic thành công!${NC}"
            # Ghi cấu hình tự động nạp ROS vào file cấu hình terminal .bashrc
            if ! grep -q "source /opt/ros/melodic/setup.bash" ~/.bashrc; then
                echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
            fi
        else
            echo -e "${RED}[ERROR] Cài đặt ROS thất bại. Hãy kiểm tra kết nối internet của xe.${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Hủy bỏ cài đặt ROS. Dừng quá trình cài đặt phần cứng.${NC}"
        exit 1
    fi
fi


# Bước 1: Thiết lập quyền truy cập cổng USB cho LiDAR (dialout group & chmod)

echo -e "\n${BLUE}[1/4] Đang thiết lập quyền cổng USB cho LiDAR...${NC}"

# Thêm user hiện tại vào group dialout để truy cập cổng serial không cần sudo
if groups $USER | grep &>/dev/null "\bdialout\b"; then
    echo -e "${GREEN}[OK] User '$USER' đã thuộc nhóm dialout.${NC}"
else
    echo -e "${YELLOW}[WARNING] Thêm user '$USER' vào nhóm dialout...${NC}"
    sudo usermod -aG dialout $USER
    echo -e "${GREEN}[SUCCESS] Đã thêm vào nhóm dialout. Vui lòng đăng nhập lại (hoặc chạy lệnh: newgrp dialout) để áp dụng.${NC}"
fi

# Cấp quyền trực tiếp cho các cổng ttyUSB nếu đang cắm
USB_DEVS=$(ls /dev/ttyUSB* 2>/dev/null)
if [ -z "$USB_DEVS" ]; then
    echo -e "${YELLOW}[WARNING] Không tìm thấy thiết bị LiDAR cổng /dev/ttyUSB* nào đang cắm vào xe.${NC}"
    echo -e "Hãy đảm bảo rằng dây USB của LiDAR đã cắm chặt vào cổng USB của Jetson Nano."
else
    for dev in $USB_DEVS; do
        echo -e "${YELLOW}Phát hiện thiết bị: $dev. Đang cấp quyền đọc/ghi (666)...${NC}"
        sudo chmod 666 "$dev"
        echo -e "${GREEN}[OK] Đã cấp quyền 666 cho $dev${NC}"
    done
fi

# Bước 2: Thiết lập udev rule cho LiDAR (Waveshare JetRacer thường dùng YDLidar)
echo -e "\n${BLUE}[2/4] Kiểm tra cấu hình Udev Rule cho LiDAR...${NC}"
UDEV_RULE_FILE="/etc/udev/rules.rules" # kiểm tra các file rules phổ biến hoặc tạo riêng
if [ -f "/etc/udev/rules.d/ydlidar.rules" ] || [ -f "/etc/udev/rules.d/rplidar.rules" ] || [ -f "/etc/udev/rules.d/99-lidar.rules" ]; then
    echo -e "${GREEN}[OK] Đã phát hiện cấu hình udev rule cho LiDAR.${NC}"
else
    echo -e "${YELLOW}Tạo cấu hình udev rule dự phòng cho cổng USB LiDAR tự động ánh xạ thành /dev/ydlidar...${NC}"
    # Tạo luật ánh xạ cho chip CP210x (YDLidar thường dùng chip này)
    sudo bash -c 'cat <<EOF > /etc/udev/rules.d/99-lidar.rules
KERNEL=="ttyUSB*", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE:="0666", SYMLINK+="ydlidar"
EOF'
    echo -e "${YELLOW}Đang tải lại luật udev...${NC}"
    sudo udevadm control --reload-rules && sudo udevadm trigger
    echo -e "${GREEN}[SUCCESS] Đã tạo luật udev. Rút LiDAR ra cắm lại để nhận cổng /dev/ydlidar${NC}"
fi

# Bước 3: Kiểm tra và Biên dịch ROS Workspace
echo -e "\n${BLUE}[3/4] Kiểm tra & biên dịch ROS Workspace...${NC}"
if [ -f "$CATKIN_WS_DIR/devel/setup.bash" ]; then
    echo -e "${GREEN}[OK] Phát hiện ROS Workspace hợp lệ tại $CATKIN_WS_DIR.${NC}"
else
    echo -e "${YELLOW}Chưa có ROS Workspace hợp lệ. Đang tiến hành tạo mới tại $HOME/catkin_ws...${NC}"
    CATKIN_WS_DIR="$HOME/catkin_ws"
    mkdir -p "$CATKIN_WS_DIR/src"
    
    echo -e "${YELLOW}Đang tải mã nguồn ROS JetRacer từ Waveshare GitHub...${NC}"
    cd "$CATKIN_WS_DIR/src"
    if [ ! -d "jetracer_ros" ] && [ ! -d "jetracer" ]; then
        git clone https://github.com/waveshare/jetracer_ros.git
        # Di chuyển các thư mục con ra ngoài nếu repo chứa các folder ROS package con
        if [ -d "jetracer_ros/jetracer" ]; then
            mv jetracer_ros/* .
            rm -rf jetracer_ros
        fi
    else
        echo -e "${GREEN}[OK] Gói jetracer đã có sẵn trong thư mục src.${NC}"
    fi
    
    # Quay lại thư mục gốc workspace để biên dịch
    cd "$CATKIN_WS_DIR"
    echo -e "${YELLOW}Đang biên dịch workspace bằng catkin_make...${NC}"
    source "$ROS_MELODIC_SETUP"
    catkin_make
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] Đã tạo và biên dịch ROS Workspace thành công tại $CATKIN_WS_DIR!${NC}"
    else
        echo -e "${RED}[ERROR] Biên dịch Workspace thất bại. Vui lòng kiểm tra lại lỗi code.${NC}"
        exit 1
    fi
fi


# Bước 4: Thiết lập cấu hình mạng ROS IP (tránh bị trễ gói tin)
echo -e "\n${BLUE}[4/4] Tối ưu hóa cấu hình mạng ROS IP...${NC}"
# Lấy IP mạng WiFi (wlan0)
WLAN_IP=$(ip addr show wlan0 2>/dev/null | grep -Eo 'inet [0-9.]+' | cut -d' ' -f2)
if [ -n "$WLAN_IP" ]; then
    echo -e "${GREEN}[OK] Phát hiện IP của xe (WiFi wlan0): $WLAN_IP${NC}"
    # Đề xuất thiết lập export vào môi trường hiện tại
    export ROS_IP="$WLAN_IP"
    export ROS_MASTER_URI="http://$WLAN_IP:11311"
    echo -e "Đã tạm thời cấu hình cho phiên chạy này:"
    echo -e "  - ROS_IP=$ROS_IP"
    echo -e "  - ROS_MASTER_URI=$ROS_MASTER_URI"
else
    echo -e "${YELLOW}[WARNING] Xe chưa kết nối WiFi (hoặc wlan0 chưa nhận IP).${NC}"
fi

echo -e "\n${GREEN}${BOLD}======================================================${NC}"
echo -e "${GREEN}${BOLD}      THIẾT LẬP PHẦN CỨNG HOÀN TẤT!                   ${NC}"
echo -e "${GREEN}${BOLD}======================================================${NC}"

# Hỏi người dùng có muốn chạy thử lidar luôn không
echo -e "\n${BOLD}Bạn có muốn chạy thử roslaunch cho LiDAR ngay bây giờ không?${NC}"
read -p "Chạy thử? (y/n): " RUN_TEST

if [[ "$RUN_TEST" =~ ^[Yy]$ ]]; then
    if [ -f "$CATKIN_WS_DIR/devel/setup.bash" ]; then
        source "$CATKIN_WS_DIR/devel/setup.bash"
        echo -e "${BLUE}Đang chạy lệnh: roslaunch jetracer lidar.launch ...${NC}"
        echo -e "${YELLOW}Nhấn Ctrl+C để tắt LiDAR và kết thúc.${NC}"
        roslaunch jetracer lidar.launch
    else
        echo -e "${RED}[ERROR] Chưa có file devel/setup.bash để nạp gói jetracer. Vui lòng kiểm tra lại catkin_make.${NC}"
    fi
else
    echo -e "Bạn có thể khởi động LiDAR thủ công sau bằng lệnh:"
    echo -e "  ${BOLD}source ~/catkin_ws/devel/setup.bash && roslaunch jetracer lidar.launch${NC}"
fi
