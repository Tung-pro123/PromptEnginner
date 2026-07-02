# 📋 Sổ Tay Cú Pháp Terminal Cho JetRacer Pro
*Tài liệu tra cứu nhanh các câu lệnh từ kết nối, khởi chạy hệ thống, cài đặt môi trường đến các script chẩn đoán.*

---

## 1. KẾT NỐI VÀ ĐĂNG NHẬP (SSH)

| Nhiệm vụ | Cú pháp lệnh / Thao tác | Ghi chú |
| :--- | :--- | :--- |
| **SSH qua cáp Micro-USB** | `ssh jetson@192.168.55.1` | Cắm cáp USB trực tiếp từ xe vào máy tính |
| **SSH qua WiFi** | `ssh jetson@<IP_WIFI_CỦA_XE>` | Máy tính và xe chung mạng WiFi |
| **Xem IP WiFi của xe** | `ip addr show wlan0` | Chạy lệnh này trên xe để lấy IP |
| **Truy cập Jupyter Notebook** | Nhập `<IP_XE>:8888` vào trình duyệt laptop | Mật khẩu mặc định thường là `jetson` |

---

## 2. QUẢN LÝ MÔI TRƯỜNG ẢO PYTHON (VENV)
*Bắt buộc thực hiện khi muốn cài đặt thêm bất kỳ thư viện Python nào để không làm hỏng driver GPU của xe.*

| Nhiệm vụ | Cú pháp lệnh | Ghi chú |
| :--- | :--- | :--- |
| **Tạo môi trường ảo** | `python3 -m venv --system-site-packages ~/my_env` | Tạo 1 lần duy nhất (nên lưu ở thư mục gốc `~/`) |
| **Kích hoạt venv** | `source ~/my_env/bin/activate` | **Cần chạy mỗi khi mở Terminal mới** |
| **Tắt môi trường ảo** | `deactivate` | Trở về môi trường Python gốc của hệ thống |

### 🛠️ Cài đặt ONNX Runtime GPU (Tối ưu cho Jetson Nano Python 3.6):
*Chạy các lệnh này bên trong môi trường ảo đã kích hoạt để cài đặt bản ONNX tăng tốc GPU CUDA:*
```bash
# 1. Tải file cài đặt (.whl) chính thức của NVIDIA cho Python 3.6
wget -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl

# 2. Cài đặt file đã tải
pip3 install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
```

---

## 3. KHỞI CHẠY HỆ THỐNG PHẦN CỨNG XE (ROS)
*Cần chạy trước khi thực hiện các bài test liên quan đến Camera, LiDAR hay Động cơ.*

### ⚠️ LƯU Ý QUAN TRỌNG VỀ MÔI TRƯỜNG ROS:
Mỗi khi mở một Tab Terminal mới trên Termius, bạn **BẮT BUỘC** phải nạp môi trường ROS bằng 2 lệnh sau:
```bash
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

| Nhiệm vụ | Cú pháp lệnh | Ghi chú |
| :--- | :--- | :--- |
| **Biên dịch code ROS** | `catkin_make` | Chạy tại thư mục gốc của ROS trên xe (`catkin_ws`) |
| **Bật phần cứng tổng hợp** | `roslaunch jetracer jetracer.launch` | Bật động cơ, IMU và Odometry EKF |
| **Bật SLAM tự lái** | `roslaunch jetracer slam_nav.launch` | Bật chế độ quét bản đồ và tự lái nâng cao |

### 💡 Khởi chạy riêng lẻ từng bộ phận (Khi cần test lẻ hoặc tránh xung đột)
*Nếu chạy code Python điều khiển động cơ trực tiếp (I2C) mà không muốn bị ROS giành quyền, ta chỉ bật Master và Cảm biến:*

| Thành phần | Cú pháp lệnh khởi chạy riêng | Ghi chú |
| :--- | :--- | :--- |
| **1. ROS Master** | `roscore` | Trung tâm điều phối (phải bật đầu tiên) |
| **2. Cảm biến LiDAR** | `roslaunch jetracer lidar.launch` | Khởi động máy quét tia laser LiDAR |
| **3. Camera CSI** | `roslaunch jetracer csi_camera.launch` | Khởi động mắt camera truyền ảnh lên ROS |

---

## 4. CHẠY CÁC SCRIPT CHẨN ĐOÁN & KIỂM TRA PHẦN CỨNG (PYTHON 3)
*Tất cả các file này nằm trong thư mục `/home/jetson/Desktop/Admin` trên xe.*

| Nhiệm vụ | Cú pháp lệnh | Ghi chú |
| :--- | :--- | :--- |
| **Test Động cơ & Lái** | `python3 tests/test_car.py` | Nhập số 0-7 để test tiến, lùi, rẽ 90 độ, căn góc lái |
| **Test Cảm biến Lẻ** | `python3 tests/test_sensors.py` | Kiểm tra FPS Camera, tần số quét LiDAR, dữ liệu góc nghiêng IMU |
| **Test LiDAR Realtime** | `python3 tests/test_only_lidar.py` | Bản đồ ASCII quét khoảng cách 6 hướng xung quanh xe |
| **Test Camera Lẻ** | `python3 tests/test_only_camera.py` | Chụp ảnh kiểm tra chất lượng thấu kính camera |
| **Test Động cơ Lẻ** | `python3 tests/test_only_motors.py` | Chỉ đánh lái trái, phải và quay tiến, lùi tốc độ chậm |
| **Chẩn đoán thư viện** | `python3 diagnostics/diagnose.py` | Kiểm tra cài đặt thư viện và thiết bị trên cổng I2C |
| **Kiểm tra xung đột** | `python3 diagnostics/inspect_jetracer.py` | Kiểm tra import thư viện jetracer sau khi lọc path |

---

## 5. CHẠY THỬ NGHIỆM THUẬT TOÁN (NÉ VẬT CẢN)

| Nhiệm vụ | Cú pháp lệnh | Ghi chú |
| :--- | :--- | :--- |
| **Test Né Vật Cản** | `python3 tests/test_obstacle_avoidance.py` | Xe chạy bám làn thẳng và tự động bẻ lái né hộp giấy |

---

## 6. MẸO VÀ LỆNH DEBUG ROS HỮU ÍCH

| Nhiệm vụ | Cú pháp lệnh | Ghi chú |
| :--- | :--- | :--- |
| **Xem danh sách node chạy** | `rosnode list` | Xem các tiến trình ROS nào đang hoạt động |
| **Xem các topic dữ liệu** | `rostopic list` | Xem các kênh truyền thông tin (ví dụ: `/scan`, `/csi_cam_0`) |
| **Xem dữ liệu thô LiDAR** | `rostopic echo /scan` | Xem luồng dữ liệu khoảng cách đang bắn về liên tục |
| **Tìm thư mục gói ROS** | `rospack find jetracer` | Xem gói ROS này đang nằm ở thư mục nào trên ổ cứng |
| **Quét tìm gói LiDAR** | `rospack list | grep -i lidar` | Tìm tên gói driver LiDAR được cài đặt trên xe |
| **Giải phóng kẹt camera** | `killall -9 python3` | Dừng khẩn cấp toàn bộ Python nếu bị kẹt camera |
