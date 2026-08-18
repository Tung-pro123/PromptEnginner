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
        
        # Tạo file log tạm thời
        ROS_LOG="/tmp/ros_install_error.log"
        > "$ROS_LOG"

        echo -e "${YELLOW}1. Cấu hình apt sources.list cho ROS...${NC}"
        if ! sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' 2>> "$ROS_LOG"; then
            echo -e "${RED}[ERROR] Cấu hình apt sources.list thất bại!${NC}"
            echo -e "${RED}Chi tiết lỗi:${NC}"
            cat "$ROS_LOG"
            exit 1
        fi
        
        echo -e "${YELLOW}2. Thiết lập khóa bảo mật apt-key...${NC}"
        sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 2>&1 | tee -a "$ROS_LOG"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            # Backup method nếu lệnh trên bị block bởi tường lửa
            curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
        fi
        
        echo -e "${YELLOW}3. Cập nhật danh sách gói hệ thống...${NC}"
        sudo apt-get update 2>&1 | tee -a "$ROS_LOG"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo -e "${RED}[ERROR] Cập nhật danh sách gói hệ thống (apt-get update) thất bại!${NC}"
            echo -e "${RED}Vui lòng kiểm tra lại kết nối internet của xe.${NC}"
            exit 1
        fi
        
        echo -e "${YELLOW}4. Sửa lỗi xung đột hệ thống (nếu có) & Cài đặt ROS Base...${NC}"
        sudo apt --fix-broken install -y
        sudo apt-get install -y ros-melodic-ros-base python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential 2>&1 | tee -a "$ROS_LOG"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo -e "${RED}[ERROR] Cài đặt các gói ROS thất bại!${NC}"
            echo -e "${RED}Chi tiết lỗi được lưu ở: $ROS_LOG${NC}"
            exit 1
        fi
        
        if ! command -v rosdep &> /dev/null; then
            echo -e "${YELLOW}Cài đặt bổ sung công cụ rosdep...${NC}"
            sudo apt-get install -y python-rosdep 2>> "$ROS_LOG" || sudo pip install -U rosdep
        fi

        if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then
            echo -e "${YELLOW}5. Khởi tạo cơ sở dữ liệu rosdep...${NC}"
            if ! sudo rosdep init 2>&1 | tee -a "$ROS_LOG"; then
                echo -e "${RED}[ERROR] Khởi tạo rosdep thất bại!${NC}"
                exit 1
            fi
        fi
        
        echo -e "${YELLOW}6. Cập nhật cơ sở dữ liệu rosdep...${NC}"
        if ! rosdep update 2>&1 | tee -a "$ROS_LOG"; then
            echo -e "${RED}[ERROR] Cập nhật rosdep thất bại!${NC}"
            exit 1
        fi
        
        ROS_SETUP_BASH=$(ls /opt/ros/*/setup.bash 2>/dev/null | head -n 1)
        if [ -n "$ROS_SETUP_BASH" ]; then
            ROS_MELODIC_SETUP="$ROS_SETUP_BASH"
            echo -e "${GREEN}[SUCCESS] Đã cài đặt ROS thành công tại $ROS_MELODIC_SETUP!${NC}"
            # Ghi cấu hình tự động nạp ROS vào file cấu hình terminal .bashrc
            if ! grep -q "source $ROS_MELODIC_SETUP" ~/.bashrc; then
                echo "source $ROS_MELODIC_SETUP" >> ~/.bashrc
            fi
        else
            echo -e "${RED}[ERROR] Không tìm thấy file setup của ROS sau khi cài đặt tại /opt/ros/.${NC}"
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
    
    echo -e "${YELLOW}Đang cài đặt các thư viện ROS phụ thuộc trước khi build...${NC}"
    sudo apt-get update

    echo -e "${YELLOW}Đang xử lý xung đột gói Python ROS (nếu có)...${NC}"
    sudo apt-get install -y --download-only python-rospkg-modules python-rosdistro-modules python-catkin-pkg-modules
    sudo dpkg -i --force-overwrite /var/cache/apt/archives/python-rospkg-modules_*.deb 2>/dev/null || true
    sudo dpkg -i --force-overwrite /var/cache/apt/archives/python-rosdistro-modules_*.deb 2>/dev/null || true
    sudo dpkg -i --force-overwrite /var/cache/apt/archives/python-catkin-pkg-modules_*.deb 2>/dev/null || true
    sudo apt --fix-broken install -y
    sudo apt-get install -y \
      ros-melodic-nav-msgs \
      ros-melodic-sensor-msgs \
      ros-melodic-geometry-msgs \
      ros-melodic-cv-bridge \
      ros-melodic-image-view \
      ros-melodic-rqt-image-view \
      ros-melodic-tf \
      ros-melodic-tf2 \
      ros-melodic-tf2-ros \
      ros-melodic-tf2-geometry-msgs \
      ros-melodic-dynamic-reconfigure \
      ros-melodic-navigation \
      ros-melodic-laser-geometry \
      ros-melodic-rplidar-ros \
      ros-melodic-joy \
      ros-melodic-teleop-twist-joy \
      ros-melodic-teleop-twist-keyboard \
      ros-melodic-gmapping \
      ros-melodic-amcl \
      ros-melodic-map-server \
      ros-melodic-move-base \
      ros-melodic-urdf \
      ros-melodic-xacro \
      ros-melodic-robot-state-publisher \
      python-catkin-pkg \
      python-rosdep \
      python-pip \
      python3-pip

    echo -e "${YELLOW}Cài đặt thư viện Python phụ thuộc (adafruit-platformdetect)...${NC}"
    sudo -H pip install adafruit-platformdetect
    sudo -H pip3 install adafruit-platformdetect

    echo -e "${YELLOW}Cài đặt catkin_pkg và rospkg cho Python 3 (sửa lỗi CMake)...${NC}"
    sudo -H pip3 install catkin_pkg rospkg empy defusedxml

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

# Hỏi người dùng có muốn chạy thử phần cứng cảm biến không
echo -e "\n${BOLD}Bạn có muốn chạy thử liên tục LiDAR (3 giây) và Camera (3 giây) không?${NC}"
read -p "Chạy thử cảm biến? (y/n): " RUN_TEST

if [[ "$RUN_TEST" =~ ^[Yy]$ ]]; then
    if [ -f "$CATKIN_WS_DIR/devel/setup.bash" ]; then
        source "$CATKIN_WS_DIR/devel/setup.bash"
        
        # 1. Đảm bảo roscore chạy ngầm nếu chưa có
        if ! pgrep -x "roscore" &>/dev/null && ! pgrep -x "rosmaster" &>/dev/null; then
            echo -e "${YELLOW}Đang khởi động ROS Master (roscore) dưới nền...${NC}"
            roscore &
            ROSCORE_PID=$!
            sleep 3
        fi
        
        # 2. Khởi chạy thử LiDAR 3 giây
        echo -e "\n${BLUE}--- [1/2] BẮT ĐẦU TEST LIDAR TRONG 3 GIÂY ---${NC}"
        roslaunch jetracer lidar.launch &
        LIDAR_PID=$!
        
        sleep 3
        
        echo -e "${RED}Đang dừng LiDAR...${NC}"
        kill $LIDAR_PID 2>/dev/null
        # Dọn dẹp tiến trình con của lidar
        killall -9 ydlidar_node rplidarNode 2>/dev/null
        sleep 1
        
        # 3. Khởi chạy thử Camera 3 giây
        echo -e "\n${BLUE}--- [2/2] BẮT ĐẦU TEST CAMERA TRONG 3 GIÂY ---${NC}"
        roslaunch jetracer csi_camera.launch &
        CAMERA_PID=$!
        
        sleep 3
        
        echo -e "${RED}Đang dừng Camera...${NC}"
        kill $CAMERA_PID 2>/dev/null
        # Dọn dẹp tiến trình con của camera
        killall -9 nvargus_daemon_client jetson_camera 2>/dev/null
        sleep 1
        
        # Dọn dẹp roscore nếu chính script này bật lên
        if [ -n "$ROSCORE_PID" ]; then
            echo -e "${RED}Đang dừng ROS Master...${NC}"
            kill $ROSCORE_PID 2>/dev/null
            killall -9 roscore rosmaster 2>/dev/null
        fi
        
        echo -e "${GREEN}${BOLD}>>> HOÀN TẤT THỬ NGHIỆM CẢM BIẾN! <<<${NC}"
    else
        echo -e "${RED}[ERROR] Chưa có file devel/setup.bash để nạp gói jetracer. Vui lòng kiểm tra lại catkin_make.${NC}"
    fi
else
    echo -e "Bạn có thể khởi động các cảm biến thủ công sau bằng lệnh:"
    echo -e "  - LiDAR:  ${BOLD}source ~/catkin_ws/devel/setup.bash && roslaunch jetracer lidar.launch${NC}"
    echo -e "  - Camera: ${BOLD}source ~/catkin_ws/devel/setup.bash && roslaunch jetracer csi_camera.launch${NC}"
fi
