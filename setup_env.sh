#!/bin/bash
# ==============================================================
# SETUP MÔI TRƯỜNG JETRACER - PromptEngineer Team
# ==============================================================
# Script này sẽ:
#   1. Kiểm tra ROS đã cài chưa
#   2. Nếu CHƯA → Tự động cài đặt ROS
#   3. Nếu RỒI → Source môi trường và sẵn sàng chạy
#
# Cách dùng:
#   chmod +x setup_env.sh
#   source ./setup_env.sh          # Chỉ nạp môi trường (nhanh)
#   bash ./setup_env.sh install    # Cài đặt ROS nếu chưa có
# ==============================================================

echo "=============================================="
echo "  🏎️  SETUP MÔI TRƯỜNG JETRACER"
echo "  Team: PromptEngineer"
echo "=============================================="

# ==============================================================
# HÀM: Cài đặt ROS
# ==============================================================
install_ros() {
    echo ""
    echo "🔧 BẮT ĐẦU CÀI ĐẶT ROS..."
    echo ""

    # Xác định phiên bản Ubuntu
    UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null)
    echo "📋 Ubuntu version: $UBUNTU_VERSION"

    if [ "$UBUNTU_VERSION" = "18.04" ]; then
        ROS_DISTRO="melodic"
    elif [ "$UBUNTU_VERSION" = "20.04" ]; then
        ROS_DISTRO="noetic"
    else
        echo "⚠️  Ubuntu $UBUNTU_VERSION - thử cài melodic (phổ biến cho Jetson Nano)"
        ROS_DISTRO="melodic"
    fi

    echo "📦 Sẽ cài đặt ROS $ROS_DISTRO..."
    echo ""

    # Bước 1: Thêm ROS repository
    echo "[1/5] Thêm ROS repository..."
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

    # Bước 2: Thêm key
    echo "[2/5] Thêm GPG key..."
    sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 2>/dev/null
    # Backup method nếu lệnh trên fail
    if [ $? -ne 0 ]; then
        curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    fi

    # Bước 3: Cập nhật package list và cài Python 3
    echo "[3/5] Cập nhật apt và cài đặt Python 3..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-dev

    # Bước 4: Cài đặt ROS
    echo "[4/5] Cài đặt ROS $ROS_DISTRO (ros-base + các package cần thiết)..."
    sudo apt-get install -y \
        ros-${ROS_DISTRO}-ros-base \
        ros-${ROS_DISTRO}-roslaunch \
        ros-${ROS_DISTRO}-sensor-msgs \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-image-transport \
        ros-${ROS_DISTRO}-gscam \
        python-rosdep \
        python-rosinstall \
        python-rosinstall-generator \
        python-wstool \
        build-essential

    # Nếu lỗi (có thể do python2 không tồn tại trên noetic)
    if [ "$ROS_DISTRO" = "noetic" ]; then
        sudo apt-get install -y \
            python3-rosdep \
            python3-rosinstall \
            python3-rosinstall-generator \
            python3-wstool
    fi

    # Bước 5: Khởi tạo rosdep
    echo "[5/5] Khởi tạo rosdep..."
    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
        sudo rosdep init 2>/dev/null
    fi
    rosdep update 2>/dev/null

    echo ""
    echo "✅ Cài đặt ROS $ROS_DISTRO hoàn tất!"
    echo ""
}

# ==============================================================
# HÀM: Tạo catkin workspace
# ==============================================================
setup_workspace() {
    echo "🔧 Tạo catkin workspace..."

    # Source ROS trước
    for distro in noetic melodic kinetic; do
        if [ -f "/opt/ros/$distro/setup.bash" ]; then
            source /opt/ros/$distro/setup.bash
            break
        fi
    done

    if [ ! -d "$HOME/catkin_ws/src" ]; then
        mkdir -p $HOME/catkin_ws/src
        echo "✅ Đã tạo workspace ~/catkin_ws"
    else
        echo "✅ Workspace ~/catkin_ws đã tồn tại"
    fi

    # Cài đặt tự động jetson_csi_cam nếu chưa có
    if [ ! -d "$HOME/catkin_ws/src/jetson_csi_cam" ]; then
        echo "📸 Đang tải package camera (jetson_csi_cam)..."
        cd $HOME/catkin_ws/src
        git clone https://github.com/peter-moran/jetson_csi_cam.git
    fi

    # Build workspace
    echo "🔨 Đang build lại catkin_ws..."
    cd $HOME/catkin_ws
    catkin_make
}

# ==============================================================
# LOGIC CHÍNH
# ==============================================================

# Nếu chạy với tham số "install" → cài đặt ROS
if [ "$1" = "install" ]; then
    # Kiểm tra đã có ROS chưa
    ROS_EXISTS=false
    for distro in noetic melodic kinetic lunar; do
        if [ -f "/opt/ros/$distro/setup.bash" ]; then
            ROS_EXISTS=true
            echo "✅ ROS $distro đã được cài đặt rồi!"
            break
        fi
    done

    if [ "$ROS_EXISTS" = false ]; then
        install_ros
        setup_workspace
    fi

    # Thêm source tự động vào .bashrc
    echo ""
    echo "🔧 Thêm tự động source vào ~/.bashrc..."

    # Tìm lại ROS distro sau khi cài
    for distro in noetic melodic kinetic; do
        if [ -f "/opt/ros/$distro/setup.bash" ]; then
            grep -q "source /opt/ros/$distro/setup.bash" ~/.bashrc 2>/dev/null
            if [ $? -ne 0 ]; then
                echo "source /opt/ros/$distro/setup.bash" >> ~/.bashrc
                echo "  ✅ Đã thêm: source /opt/ros/$distro/setup.bash"
            fi
            break
        fi
    done

    # Tìm workspace
    for ws_name in catkin_ws jetracer_ws ros_ws jetbot_ws; do
        if [ -f "$HOME/$ws_name/devel/setup.bash" ]; then
            grep -q "source.*$ws_name/devel/setup.bash" ~/.bashrc 2>/dev/null
            if [ $? -ne 0 ]; then
                echo "source $HOME/$ws_name/devel/setup.bash" >> ~/.bashrc
                echo "  ✅ Đã thêm: source ~/$ws_name/devel/setup.bash"
            fi
            break
        fi
    done

    echo ""
    echo "🎉 CÀI ĐẶT HOÀN TẤT! Chạy lại: source ~/.bashrc"
    echo ""
    return 0 2>/dev/null || exit 0
fi

# ==============================================================
# CHẾ ĐỘ MẶC ĐỊNH: Chỉ nạp môi trường (source)
# ==============================================================

# --- Tìm và source ROS ---
ROS_FOUND=false

# Ưu tiên: ROS trên ổ ngoài (Jetson boot từ USB/SSD)
for f in /media/jetson/*/opt/ros/*/setup.bash; do
    if [ -f "$f" ]; then
        distro_name=$(basename $(dirname "$f"))
        echo "✅ ROS $distro_name (external drive)"
        source "$f"
        ROS_FOUND=true
        break
    fi
done

# Fallback: ROS ở vị trí tiêu chuẩn
if [ "$ROS_FOUND" = false ]; then
    for distro in noetic melodic kinetic lunar; do
        if [ -f "/opt/ros/$distro/setup.bash" ]; then
            echo "✅ ROS $distro"
            source /opt/ros/$distro/setup.bash
            ROS_FOUND=true
            break
        fi
    done
fi

# Thử tìm ở vị trí khác
if [ "$ROS_FOUND" = false ]; then
    for f in /opt/ros/*/setup.bash; do
        if [ -f "$f" ]; then
            echo "✅ ROS ($(basename $(dirname $f)))"
            source "$f"
            ROS_FOUND=true
            break
        fi
    done
fi

if [ "$ROS_FOUND" = false ]; then
    echo ""
    echo "❌ CHƯA CÀI ĐẶT ROS!"
    echo ""
    echo "👉 Để cài đặt ROS, chạy lệnh sau:"
    echo "   bash ~/setup_env.sh install"
    echo ""
    return 1 2>/dev/null || exit 1
fi

# --- Tìm và source catkin workspace ---
WS_FOUND=false
for ws_name in catkin_ws jetracer_ws ros_ws workspace jetbot_ws; do
    if [ -f "$HOME/$ws_name/devel/setup.bash" ]; then
        echo "✅ Workspace: ~/$ws_name"
        source "$HOME/$ws_name/devel/setup.bash"
        WS_FOUND=true
        break
    fi
done

if [ "$WS_FOUND" = false ]; then
    for d in $HOME/*/devel/setup.bash; do
        if [ -f "$d" ]; then
            ws_path=$(dirname $(dirname "$d"))
            echo "✅ Workspace: $(basename $ws_path)"
            source "$d"
            WS_FOUND=true
            break
        fi
    done
fi

if [ "$WS_FOUND" = false ]; then
    echo "⚠️  Chưa tìm thấy catkin workspace"
fi

# --- Kiểm tra roslaunch ---
if command -v roslaunch &> /dev/null; then
    echo "✅ roslaunch sẵn sàng!"
else
    echo "❌ roslaunch không tìm thấy"
    echo "   Chạy: bash ~/setup_env.sh install"
    return 1 2>/dev/null || exit 1
fi

# --- Tìm package jetracer ---
JETRACER_PKG="jetracer"
for pkg in jetracer jetracer_pro jetbot_pro jetracer_ros; do
    if rospack find "$pkg" &> /dev/null; then
        JETRACER_PKG="$pkg"
        echo "✅ Package: $JETRACER_PKG"
        break
    fi
done

# --- Hướng dẫn ---
echo ""
echo "=============================================="
echo "  🎉 SẴN SÀNG!"
echo "=============================================="
echo ""
echo "  Tab 1 - Bật LiDAR:"
echo "     roslaunch $JETRACER_PKG lidar.launch"
echo ""
echo "  Tab 2 - Bật Camera:"
echo "     roslaunch $JETRACER_PKG csi_camera.launch"
echo ""
echo "  Tab 3 - Speed Track:"
echo "     cd ~/hackathon-code-base"
echo "     python3 src/speed_track/main_speed_track.py"
echo ""
echo "=============================================="
