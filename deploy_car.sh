#!/bin/bash
# ============================================================
# TỰ ĐỘNG CÀI ĐẶT MÔI TRƯỜNG ĐUA XE (OFFLINE/ONLINE)
# Dành cho xe Jetson Nano thi đấu
# ============================================================

echo "=================================================="
echo "🚀 BẮT ĐẦU CÀI ĐẶT XE THI ĐẤU"
echo "=================================================="

# 1. Bẻ khóa quyền điều khiển phần cứng ngay lập tức
echo "[1/4] Mở khóa quyền phần cứng (Motor/Servo)..."
sudo chmod a+rw /dev/i2c-* /dev/gpiochip* 2>/dev/null
echo "✅ Đã cấp quyền phần cứng."

# 2. Cài đặt các thư viện hệ thống cần thiết (Bao gồm LiDAR ROS)
echo "[2/4] Kiểm tra Internet để tải thư viện hệ thống (ROS, Lidar, OpenCV)..."
wget -q --spider http://google.com
if [ $? -eq 0 ]; then
    echo "Có Internet! Đang tải thư viện..."
    sudo apt-get update -y
    sudo apt-get install -y python3-pip python3-opencv ros-melodic-gscam python3-rospkg python3-catkin-pkg ros-melodic-rplidar-ros ros-melodic-urg-node
    echo "✅ Đã cài đặt thư viện hệ thống và LiDAR ROS."
else
    echo "⚠️ BỎ QUA: Không có Internet. Sẽ sử dụng thư viện hệ thống đã cài sẵn trên xe!"
fi

# 3. Cài đặt Python Dependencies (Từ thư mục offline nếu có)
echo "[3/4] Cài đặt Python Dependencies..."
if [ -d "offline_libs" ]; then
    echo "Phát hiện thư mục offline_libs! Đang cài đặt không cần mạng..."
    pip3 install --no-index --find-links=offline_libs -r requirements_jetson.txt
else
    echo "Đang cài đặt qua Internet..."
    pip3 install -r requirements_jetson.txt
fi
echo "✅ Đã cài đặt thư viện Motor (Adafruit)."

# 4. Vá lỗi mã nguồn (OpenCV và VideoWriter)
echo "[4/4] Vá lỗi mã nguồn tương thích tự động..."
python3 -c "
import os
files_to_patch = ['src/speed_track/main_speed_simple.py', 'src/speed_track/main_speed_competition.py']
for file_path in files_to_patch:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Patch 1: OpenCV findContours
        if 'contours, _ = cv2.findContours' in code:
            code = code.replace('contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)', 'contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]')
            
        # Patch 2: OpenCV VideoWriter resize
        if 'self.video_writer.write(frame)' in code and 'cv2.resize' not in code:
            code = code.replace('self.video_writer.write(frame)', 'if frame.shape[0] != self.H or frame.shape[1] != self.W:\n                frame = cv2.resize(frame, (self.W, self.H))\n            self.video_writer.write(frame)')
            
        with open(file_path, 'w') as f:
            f.write(code)
"
echo "✅ Đã vá lỗi thành công."

echo "=================================================="
echo "🎉 HOÀN TẤT! XE ĐÃ SẴN SÀNG THI ĐẤU."
echo "Bạn có thể chạy thử lệnh:"
echo "python3 src/speed_track/main_speed_competition.py"
echo "=================================================="
