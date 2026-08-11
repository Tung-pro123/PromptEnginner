#!/bin/bash

# ==============================================================================
# Jetson AI Racer - Tự động dọn dẹp và Build lại ROS Workspace (catkin_ws)
# ==============================================================================

# Màu sắc
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}======================================================${NC}"
echo -e "${BLUE}${BOLD}        TỰ ĐỘNG REBUILD ROS WORKSPACE (CATKIN_WS)     ${NC}"
echo -e "${BLUE}${BOLD}======================================================${NC}"

# 1. Tìm thư mục cài đặt ROS
ROS_SETUP_BASH=$(ls /opt/ros/*/setup.bash 2>/dev/null | head -n 1)
if [ -z "$ROS_SETUP_BASH" ]; then
    echo -e "${RED}[ERROR] Không tìm thấy bản cài đặt ROS nào trên hệ thống (/opt/ros/*).${NC}"
    echo -e "Vui lòng cài đặt ROS trước khi build workspace."
    exit 1
fi
echo -e "${GREEN}[OK] Phát hiện ROS script tại: $ROS_SETUP_BASH${NC}"
source "$ROS_SETUP_BASH"

# 2. Tìm thư mục catkin_ws
SETUP_BASH_PATH=$(find "$HOME" -maxdepth 3 -name "setup.bash" | grep "devel/setup.bash" | head -n 1)
if [ -n "$SETUP_BASH_PATH" ]; then
    CATKIN_WS_DIR=$(dirname "$(dirname "$SETUP_BASH_PATH")")
else
    # Nếu không tìm thấy, mặc định là ~/catkin_ws
    CATKIN_WS_DIR="$HOME/catkin_ws"
fi

if [ ! -d "$CATKIN_WS_DIR/src" ]; then
    echo -e "${RED}[ERROR] Không tìm thấy thư mục src trong workspace: $CATKIN_WS_DIR${NC}"
    echo -e "${YELLOW}Workspace có vẻ không tồn tại hoặc chưa hợp lệ.${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] Sử dụng ROS Workspace tại: $CATKIN_WS_DIR${NC}"
cd "$CATKIN_WS_DIR"

# 3. Dọn dẹp thư mục build và devel cũ
echo -e "\n${YELLOW}Đang dọn dẹp (clean) workspace cũ...${NC}"
rm -rf build/ devel/ install/
echo -e "${GREEN}[OK] Đã xóa thư mục build/, devel/ và install/ (nếu có).${NC}"

# 4. Cài đặt các thư viện ROS phụ thuộc
echo -e "\n${YELLOW}Đang cài đặt các thư viện ROS phụ thuộc trước khi build...${NC}"
sudo apt-get update
sudo apt-get install -y \
  ros-melodic-nav-msgs \
  ros-melodic-sensor-msgs \
  ros-melodic-geometry-msgs \
  ros-melodic-tf \
  ros-melodic-tf2 \
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

# 5. Biên dịch lại workspace
echo -e "\n${YELLOW}Đang tiến hành biên dịch với catkin_make...${NC}"
catkin_make

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}======================================================${NC}"
    echo -e "${GREEN}${BOLD} [SUCCESS] BIÊN DỊCH THÀNH CÔNG ROS WORKSPACE!        ${NC}"
    echo -e "${GREEN}${BOLD}======================================================${NC}"
    
    # Tự động nạp môi trường mới
    source "$CATKIN_WS_DIR/devel/setup.bash"
    
    # Gợi ý cho người dùng thêm vào ~/.bashrc nếu chưa có
    if ! grep -q "source $CATKIN_WS_DIR/devel/setup.bash" ~/.bashrc; then
        echo -e "\n${YELLOW}Ghi chú: Có vẻ bạn chưa tự động nạp workspace này vào ~/.bashrc.${NC}"
        echo -e "Để không cần chạy lệnh source ở các terminal mới, bạn có thể chạy:"
        echo -e "${BOLD}echo \"source $CATKIN_WS_DIR/devel/setup.bash\" >> ~/.bashrc${NC}"
    fi
    
    echo -e "\n${BLUE}Để cập nhật môi trường cho terminal hiện tại, hãy chạy:${NC}"
    echo -e "${BOLD}source $CATKIN_WS_DIR/devel/setup.bash${NC}"
else
    echo -e "\n${RED}${BOLD}======================================================${NC}"
    echo -e "${RED}${BOLD} [ERROR] BIÊN DỊCH THẤT BẠI. VUI LÒNG KIỂM TRA LỖI CODE!${NC}"
    echo -e "${RED}${BOLD}======================================================${NC}"
    exit 1
fi
