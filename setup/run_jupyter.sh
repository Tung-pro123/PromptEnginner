#!/bin/bash
# Script khởi chạy Jupyter Notebook Server trên Jetson Nano để kết nối từ Laptop cá nhân

# Màu sắc giao diện hiển thị
GREEN='\033[0;32m'
NC='\033[0m' # No Color
YELLOW='\033[1;33m'
BLUE='\033[0;34m'

echo -e "${BLUE}=== KIỂM TRA VÀ KHỞI CHẠY JUPYTER NOTEBOOK SERVER ===${NC}"

# 1. Kiểm tra xem jupyter đã được cài đặt chưa
if ! command -v jupyter &> /dev/null
then
    echo -e "${YELLOW}[WARN] Jupyter Notebook chưa được cài đặt trên thiết bị.${NC}"
    echo -e "Đang tiến hành cài đặt jupyter bằng pip..."
    pip3 install jupyter
else
    echo -e "${GREEN}[OK] Jupyter Notebook đã được cài đặt.${NC}"
fi

# 2. Tìm địa chỉ IP của Jetson Nano trong mạng cục bộ (LAN/WiFi)
JETSON_IP=$(hostname -I | awk '{print $1}')

if [ -z "$JETSON_IP" ]; then
    echo -e "${YELLOW}[WARN] Không tìm thấy kết nối mạng của Jetson Nano. Đang gán IP mặc định là 0.0.0.0${NC}"
    JETSON_IP="0.0.0.0"
fi

echo -e "\n${BLUE}=================== HƯỚNG DẪN KẾT NỐI ===================${NC}"
echo -e "1. Đảm bảo Laptop của bạn và Jetson Nano ${YELLOW}dùng chung mạng WiFi/LAN${NC}."
echo -e "2. Sau khi Jupyter khởi động xong, màn hình sẽ hiển thị link có dạng:"
echo -e "   ${GREEN}http://127.0.0.1:8888/?token=xxxxxxxxxxxxxxxxxxxxxxx${NC}"
echo -e "3. Hãy copy link đó, dán lên trình duyệt của Laptop, và sửa:"
echo -e "   ${YELLOW}127.0.0.1${NC} thành địa chỉ IP của Jetson Nano: ${GREEN}${JETSON_IP}${NC}"
echo -e "   Ví dụ: ${BLUE}http://${JETSON_IP}:8888/?token=xxxxxxxxxxxxxxxxxxxxxxx${NC}"
echo -e "${BLUE}========================================================${NC}\n"

# 3. Khởi chạy Jupyter Server cho phép truy cập từ bên ngoài
# --ip=0.0.0.0: Cho phép nhận kết nối từ mọi thiết bị trong mạng
# --no-browser: Không tự động mở trình duyệt trên Jetson (vì đang dùng dạng headless/SSH)
# --allow-root: Cho phép chạy bằng quyền root (nếu cần thiết)
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
