# Jetson AI Racer Challenge 2026 🏎️🤖
## Team: PromptEngineer

> **Cuộc thi:** Jetson AI Racer Challenge 2026 - FPT Education  
> **Nền tảng:** NVIDIA Jetson Nano + JetRacer/JetBot  
> **Ngôn ngữ:** Python 3 + ROS (Robot Operating System)

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
Jetson/
├── diagnostics/                   # 🔍 Các script chẩn đoán phần cứng hệ thống (Python 3)
│   ├── diagnose.py               #   Kiểm tra I2C (quét cổng 0x40), kiểm tra các thư viện điều khiển gốc
│   └── inspect_jetracer.py       #   Kiểm tra import thư viện jetracer sau khi lọc path Python 2
│
├── docs/                          # 📄 Tài liệu hướng dẫn & thể lệ cuộc thi
│   ├── Thể lệ.docx.pdf           #   Thể lệ & luật thi chính thức
│   ├── Đề bài chi tiết.docx.pdf   #   Đề bài chi tiết (Speed Track + Smart City)
│   ├── TERMINAL_COMMANDS.md      #   Sổ tay tra cứu nhanh các câu lệnh Terminal trên xe
│   └── SPEED_TRACK_ALGORITHM.md  #   Tài liệu giải thích thuật toán điều khiển LQR & né vật cản
│
├── src/                           # 🧑‍💻 SOURCE CODE CHÍNH (code thuật toán ở đây!)
│   ├── core/                      #   🔧 Module lõi dùng chung cho cả 2 bài thi
│   │   ├── perception/            #     👁️ Xử lý cảm biến (Dual-ROI camera, LiDAR)
│   │   ├── control/               #     🎮 Điều khiển động cơ xe
│   │   │   ├── racer_controller.py #      Bộ điều khiển động cơ gốc (Hỗ trợ Ackermann & JetBot)
│   │   │   └── lqr_controller.py  #       Bộ điều khiển LQR bám làn & Tránh vật cản (S-Curve offset)
│   │   ├── planning/              #     🗺️ Điều hướng & tìm đường (A*, Dijkstra)
│   │   └── utils/                 #     🛠️ Tiện ích & dữ liệu
│   │
│   ├── speed_track/               #   🏁 BÀI THI 1: SPEED TRACK (30% điểm)
│   │   └── main_speed_track.py    #     File chạy chính bám làn tốc độ cao & né vật cản
│   │
│   └── smart_city/                #   🏙️ BÀI THI 2: SMART CITY (40% điểm)
│       └── main_smart_city.py     #     File chạy chính - FSM biển báo, QR code, điều hướng giao lộ
│
├── tests/                         # 🧪 CÁC FILE TEST VÀ CÂN CHỈNH RIÊNG BIỆT (Gọn gàng & Cách ly)
│   ├── test_car.py               #   Kiểm tra ga/lái, cân chỉnh lái thẳng & rẽ 90 độ
│   ├── test_obstacle_avoidance.py #   Chạy thử nghiệm thuật toán né vật cản đường thẳng (LQR)
│   ├── test_only_camera.py       #   Test camera riêng lẻ (Lưu chuỗi ảnh không ghi đè vào captured_images/)
│   ├── test_only_lidar.py        #   Bản đồ ASCII theo dõi khoảng cách LiDAR thời gian thực
│   ├── test_only_motors.py       #   Test cơ cấu đánh lái trái/phải và ga độc lập
│   ├── test_path_ordering.py     #   Kiểm tra thứ tự đường dẫn sys.path của Python 2 & Python 3
│   └── test_sensors.py           #   Quét báo cáo sức khỏe tổng quan của tất cả cảm biến (Cam/LiDAR/IMU)
│
├── captured_images/               # 📸 Thư mục tự động tạo chứa ảnh chụp từ test_only_camera.py
├── archive/                       # 📦 Code backup cũ từ Hackathon (KHÔNG SỬA Ở ĐÂY)
├── .gitignore                     # Git ignore các file tạm, ảnh test, môi trường ảo venv
└── README.md                      # 📖 File này (Hướng dẫn tổng quan cho thành viên)
```

---

## 🏁 Tóm tắt 2 Bài Thi

### Bài 1: Speed Track (30% điểm)
* **File chạy chính:** `src/speed_track/main_speed_track.py`
* **Thuật toán cốt lõi:** Bám làn đường thẳng/cong bằng bộ điều khiển tối ưu **LQR Controller** kết hợp cảm biến **LiDAR** phát hiện vật cản để dịch vạch ảo (**Offset S-Curve**) giúp xe lách tránh mượt mà.

### Bài 2: Smart City (40% điểm)
* **File chạy chính:** `src/smart_city/main_smart_city.py`
* **Thuật toán cốt lõi:** Di chuyển ngắn nhất theo đồ thị (A* / Dijkstra), quản lý hành vi tại giao lộ bằng **FSM (Finite State Machine)**, nhận diện biển báo giao thông thời gian thực qua **Roboflow API/YOLO**.

---

## 🚀 Hướng Dẫn Thiết Lập & Chạy Nhanh Cho Thành Viên

### 1. Tạo Môi Trường Ảo (Chạy 1 lần duy nhất khi mượn xe mới)
Tránh cài thư viện trực tiếp vào hệ thống của xe để không làm lỗi OpenCV và driver GPU gốc:
```bash
# Tạo môi trường ảo thừa hưởng thư viện hệ thống (OpenCV, CUDA, ROS)
python3 -m venv --system-site-packages ~/my_env

# Kích hoạt môi trường ảo (Cần chạy mỗi khi mở Terminal mới)
source ~/my_env/bin/activate
```

### 2. Cài Đặt ONNX Runtime GPU (Tải bản build riêng của NVIDIA)
Vì xe chạy Python 3.6 và chip ARM64, hãy chạy lệnh này trong venv để cài đặt nhanh bản GPU:
```bash
# 1. Tải file .whl chuẩn
wget -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl

# 2. Tiến hành cài đặt
pip3 install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
```

### 3. Quy Trình Tránh Xung Đột Động Cơ (Không Chạy Lệnh Tổng Khi Debug)
Do code Python (`RacerController`) và ROS node `jetracer` đều ghi vào cổng I2C điều khiển động cơ nên sẽ gây xung đột khóa bánh. Khi muốn chạy code của chúng ta kết hợp cảm biến, hãy làm như sau:

* **Terminal 1:** Chỉ bật trung tâm điều phối:
  ```bash
  roscore
  ```
* **Terminal 2:** Nạp môi trường ROS và bật cảm biến:
  ```bash
  source /opt/ros/melodic/setup.bash
  source ~/catkin_ws/devel/setup.bash
  # Bật LiDAR & Camera
  roslaunch jetracer lidar.launch
  roslaunch jetracer csi_camera.launch
  ```
* **Terminal 3:** Kích hoạt môi trường ảo và chạy code thuật toán:
  ```bash
  source ~/my_env/bin/activate
  cd ~/Desktop/Admin
  # Ví dụ chạy file test né vật cản
  python3 tests/test_obstacle_avoidance.py
  ```

---

## 🧪 Hướng Dẫn Cân Chỉnh Xe (Calibrate)

Các thành viên phụ trách phần **Control** cần chạy các script test sau để tinh chỉnh tham số phần cứng:

1. **Căn chỉnh lái thẳng & góc rẽ 90 độ:**
   ```bash
   python3 tests/test_car.py
   ```
   * Chọn Option 1, 3, 4 để kiểm tra xe đi thẳng có bị lệch không. Nếu lệch, chỉnh sửa giá trị `STEERING_OFFSET` trong file cấu hình `src/core/utils/config.py` (hoặc `.json`).
   * Chọn Option 5, 6 để test rẽ góc. Chỉnh sửa `STEERING_VALUE_FOR_TURN` và `TURN_DURATION_90_DEG` cho tới khi xe ôm cua tròn đúng 90 độ.

2. **Kiểm tra mắt đọc cảm biến:**
   ```bash
   # Quét sức khỏe toàn bộ cảm biến
   python3 tests/test_sensors.py
   
   # Theo dõi khoảng cách LiDAR thời gian thực dưới dạng ASCII trực quan
   python3 tests/test_only_lidar.py
   ```

3. **Chụp ảnh test camera:**
   ```bash
   python3 tests/test_only_camera.py
   ```
   * File ảnh chụp sẽ được lưu tự động dạng `frame_direct_1.jpg`, `frame_direct_2.jpg`,... trong thư mục `captured_images/` mà không lo bị đè ảnh cũ.

---

## ⚠️ Lưu Ý Quan Trọng Cho Cả Đội

1. **Nạp môi trường ROS:** Mỗi khi mở tab Terminal mới, luôn nhớ gõ `source ~/catkin_ws/devel/setup.bash` trước khi chạy các lệnh ROS.
2. **Quản lý file test:** Vui lòng không tạo thêm các file test lẻ tẻ ngoài thư mục gốc. Hãy cho tất cả vào thư mục `tests/` và kế thừa cách import thư mục gốc bằng `sys.path`.
3. **Pin xe:** Khi pin sụt dưới 11V, tốc độ động cơ sẽ bị yếu đi và góc lái servo có thể bị trễ. Luôn đo pin trước khi calibrate hệ số PID/LQR.
4. **Tài liệu tra cứu:** Mở file [TERMINAL_COMMANDS.md](file:///d:/Jetson/Jetson/docs/TERMINAL_COMMANDS.md) trong thư mục `docs/` để xem tất cả các câu lệnh mẫu khi lên sân thi.

---
*Chúc đội thi PromptEngineer có một mùa giải thành công rực rỡ! 🚀🏆*
