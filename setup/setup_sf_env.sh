#!/bin/bash

# ==============================================================================
# Jetson AI Racer Challenge 2026 - Environment Setup Script
# Đội: PromptEngineer
# ==============================================================================

# Cấu hình đường dẫn
VENV_DIR="$HOME/my_env"
WHL_URL="https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl"
WHL_FILE="onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl"

# Màu sắc hiển thị
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}======================================================${NC}"
echo -e "${BLUE}${BOLD}     MÔI TRƯỜNG CÀI ĐẶT SETUP - PROMPTENGINEER        ${NC}"
echo -e "${BLUE}${BOLD}======================================================${NC}"

# Bước 1: Kiểm tra và cài đặt Python 3 cùng các thư viện hệ thống thiết yếu
echo -e "\n${BLUE}[1/6] Kiểm tra & cài đặt Python 3 và các gói hệ thống cần thiết...${NC}"

# Hàm kiểm tra và cài gói
install_system_package() {
    PACKAGE=$1
    if dpkg -s "$PACKAGE" &>/dev/null; then
        echo -e "${GREEN}[OK] Gói hệ thống '$PACKAGE' đã được cài đặt.${NC}"
    else
        echo -e "${YELLOW}[WARNING] Không tìm thấy gói '$PACKAGE'. Tiến hành cài đặt...${NC}"
        sudo apt-get update && sudo apt-get install -y "$PACKAGE"
        if [ $? -ne 0 ]; then
            echo -e "${RED}[ERROR] Không thể cài đặt gói '$PACKAGE'. Vui lòng chạy lệnh sau thủ công:${NC}"
            echo -e "sudo apt-get update && sudo apt-get install -y $PACKAGE"
            exit 1
        fi
    fi
}

# Kiểm tra python3
if command -v python3 &>/dev/null; then
    echo -e "${GREEN}[OK] Python 3 đã được cài đặt: $(python3 --version)${NC}"
else
    echo -e "${YELLOW}[WARNING] Không tìm thấy Python 3! Tiến hành cài đặt...${NC}"
    sudo apt-get update && sudo apt-get install -y python3
fi

# Cài đặt các gói dev, pip và venv cho python3
install_system_package "python3-dev"
install_system_package "python3-pip"
install_system_package "python3-venv"

# Kiểm tra và đảm bảo lệnh pip3 hoạt động (Dự phòng nếu python3-pip bị thiếu liên kết)
if command -v pip3 &>/dev/null; then
    echo -e "${GREEN}[OK] Lệnh pip3 hoạt động bình thường: $(pip3 --version)${NC}"
else
    echo -e "${YELLOW}[WARNING] Không tìm thấy lệnh pip3. Đang tải và cài đặt bằng get-pip.py (phiên bản cho Python 3.6)...${NC}"
    wget https://bootstrap.pypa.io/pip/3.6/get-pip.py -O get-pip.py || curl -sSL https://bootstrap.pypa.io/pip/3.6/get-pip.py -o get-pip.py
    if [ -f get-pip.py ]; then
        python3 get-pip.py
        rm -f get-pip.py
    else
        echo -e "${RED}[ERROR] Không thể tải get-pip.py. Vui lòng kiểm tra kết nối Internet.${NC}"
        exit 1
    fi
fi


# Bước 2: Tạo Virtual Environment với system-site-packages
echo -e "\n${BLUE}[2/6] Tạo môi trường ảo Python 3 (thừa hưởng thư viện hệ thống)...${NC}"
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo -e "${GREEN}[OK] Môi trường ảo đã tồn tại tại $VENV_DIR.${NC}"
else
    echo -e "${YELLOW}Chưa có môi trường ảo hợp lệ tại $VENV_DIR. Đang tạo mới...${NC}"
    # Đảm bảo thư mục cha tồn tại
    mkdir -p "$(dirname "$VENV_DIR")"
    python3 -m venv --system-site-packages "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Tạo môi trường ảo thất bại. Thử lại với python3-venv...${NC}"
        sudo apt-get install -y python3-venv && python3 -m venv --system-site-packages "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo -e "${RED}[ERROR] Không thể tạo môi trường ảo.${NC}"
            exit 1
        fi
    fi
    echo -e "${GREEN}[SUCCESS] Đã tạo môi trường ảo tại $VENV_DIR${NC}"
fi

# Kích hoạt venv (cho phiên shell hiện tại của script)
source "$VENV_DIR/bin/activate"

# Bước 3: Nâng cấp pip và cài đặt wheel tool trong venv
echo -e "\n${BLUE}[3/6] Cập nhật pip và các công cụ cơ bản trong venv...${NC}"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# Bước 4: Tải và cài đặt ONNX Runtime GPU cho Jetson Nano
echo -e "\n${BLUE}[4/6] Cài đặt ONNX Runtime GPU (CUDA tăng tốc)...${NC}"
if "$VENV_DIR/bin/python3" -c "import onnxruntime" &>/dev/null; then
    echo -e "${GREEN}[INFO] ONNX Runtime đã được cài đặt trong môi trường.${NC}"
else
    if [ ! -f "$WHL_FILE" ]; then
        echo -e "${YELLOW}Đang tải ONNX Runtime GPU từ NVIDIA Box...${NC}"
        wget -O "$WHL_FILE" "$WHL_URL"
        if [ $? -ne 0 ]; then
            echo -e "${RED}[ERROR] Không thể tải file .whl từ máy chủ NVIDIA. Hãy kiểm tra kết nối mạng!${NC}"
            exit 1
        fi
    fi

    echo -e "${YELLOW}Đang tiến hành cài đặt file wheel vào venv...${NC}"
    "$VENV_DIR/bin/pip" install "$WHL_FILE"
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Cài đặt ONNX Runtime GPU thất bại.${NC}"
        exit 1
    fi
    echo -e "${GREEN}[SUCCESS] Cài đặt ONNX Runtime GPU thành công!${NC}"
    # Xóa file cài đặt sau khi hoàn tất để tiết kiệm dung lượng bộ nhớ xe
    rm -f "$WHL_FILE"
fi

# Bước 5: Cài đặt các thư viện bổ trợ cần thiết khác vào venv
echo -e "\n${BLUE}[5/6] Cài đặt các thư viện Python bổ trợ...${NC}"
# Sử dụng các gói thông dụng cho xử lý LiDAR, lọc dữ liệu (filterpy), v.v.
"$VENV_DIR/bin/pip" install numpy pyyaml filterpy pyserial

# Bước 6: Kiểm tra và xác thực cài đặt
echo -e "\n${BLUE}[6/6] Đang kiểm tra tích hợp hệ thống...${NC}"
echo -e "${BOLD}Thông tin môi trường ảo hiện tại:${NC}"
echo -e "  - Đường dẫn Python venv: $VENV_DIR/bin/python3"
echo -e "  - Phiên bản Python: $($VENV_DIR/bin/python3 --version)"

# Kiểm tra OpenCV
echo -n "  - Trạng thái OpenCV: "
"$VENV_DIR/bin/python3" -c "import cv2; print('OK (Phiên bản: ' + cv2.__version__ + ')')" 2>/dev/null || echo -e "${RED}Lỗi: Không tìm thấy OpenCV trong venv${NC}"

# Kiểm tra ONNX Runtime
echo -n "  - Trạng thái ONNX Runtime: "
"$VENV_DIR/bin/python3" -c "import onnxruntime; print('OK (Providers: ' + str(onnxruntime.get_available_providers()) + ')')" 2>/dev/null || echo -e "${RED}Lỗi: Không tìm thấy ONNX Runtime trong venv${NC}"

# Kiểm tra NumPy
echo -n "  - Trạng thái NumPy: "
"$VENV_DIR/bin/python3" -c "import numpy; print('OK (Phiên bản: ' + numpy.__version__ + ')')" 2>/dev/null || echo -e "${RED}Lỗi: Không tìm thấy NumPy trong venv${NC}"

echo -e "\n${GREEN}${BOLD}======================================================${NC}"
echo -e "${GREEN}${BOLD}     CÀI ĐẶT HOÀN TẤT! SẴN SÀNG LÊN SÂN ĐẤU!          ${NC}"
echo -e "${GREEN}${BOLD}======================================================${NC}"
echo -e "Để kích hoạt môi trường ảo mỗi khi chạy, dùng lệnh:"
echo -e "  ${BOLD}source ~/my_env/bin/activate${NC}"
