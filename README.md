# Jetson AI Racer Challenge 2026 🏎️🤖
## Team: PromptEngineer

> **Cuộc thi:** Jetson AI Racer Challenge 2026 - FPT Education  
> **Nền tảng:** NVIDIA Jetson Nano + Waveshare JetRacer Pro AI Kit (Ackermann Steering)  
> **Ngôn ngữ:** Python 3 + ROS (Robot Operating System) + PyTorch / ONNX Runtime

---

## 📁 Cấu Trúc Thư Mục Chuẩn Hóa (Clean Modular Architecture)

```
robot-jeston/
├── docs/                           # 📄 TÀI LIỆU TOÀN DIỆN
│   ├── contest/                    #   Đề bài, Thể lệ, Proposal & file trích xuất
│   ├── papers/                     #   Các bài báo khoa học & tài liệu nghiên cứu xe tự hành
│   ├── architecture/               #   Tài liệu kiến trúc hệ thống, luồng dữ liệu & blackboard
│   ├── algorithms/                 #   Sổ tay giải thuật LQR, S-Curve dodging, Calibration
│   └── terminal/                   #   Sổ tay tra cứu câu lệnh Terminal trên xe
│
├── robot/                          # 🧑‍💻 THƯ VIỆN LÕI DÙNG CHUNG (Shared Core Library)
│   ├── config/                     #   Cài đặt phần cứng, camera, ma trận BEV, PID/LQR gains
│   ├── perception/                 #   Xử lý cảm biến: BEV, MultiLaneDetector V3, LiDAR, Traffic/Sign
│   ├── estimation/                 #   Ước lượng trạng thái: LaneStateEstimator, Geometry engine
│   ├── planning/                   #   Định tuyến & tìm đường: MapNavigator (A*, Dijkstra)
│   ├── control/                    #   Bộ điều khiển chấp hành: RacerController, LQR, Stanley, PurePursuit
│   ├── fsm/                        #   Máy trạng thái hữu hạn né vật cản (FSM Manager)
│   ├── ai/                         #   Bộ não ra quyết định phân cấp (AI Decision Engine)
│   ├── dagger/                     #   Học bắt chước trực tuyến DAgger 15D (State, Policy, ReplayBuffer)
│   ├── debug/                      #   Công cụ gỡ lỗi (Visualizer, Tune HSV, Logger, Debugger)
│   └── utils/                      #   Tiện ích chung (Blackboard, CSV logger, Error logger)
│
├── runners/                        # 🏁 CÁC CHƯƠNG TRÌNH THỰC THI (Entrypoints)
│   ├── speed_track/                #   🏁 Bài thi 1: Speed Track (main_speed_track_v3.py)
│   ├── smart_city/                 #   🏙️ Bài thi 2: Smart City (main_smart_city.py)
│   ├── dagger/                     #   🎮 Bài học bắt chước: DAgger Joy (main_dagger.py)
│   ├── teleop/                     #   🕹️ Lái xe thủ công bằng Gamepad (ros_joy.py)
│   └── navigation/                 #   🧭 Điều hướng tự hành mở rộng (ros_ai.py)
│
├── training/                       # 🏋️ HUẤN LUYỆN OFFLINE & XUẤT MÔ HÌNH
│   ├── train_dagger_offline.py     #   Train DAgger từ logs CSV với Left-Right Mirror Augmentation
│   ├── export_dagger.py            #   Xuất DAgger Policy PyTorch sang ONNX
│   ├── extract_frames.py           #   Trích xuất khung hình từ video log
│   └── image_segmentation.py       #   Công cụ phân đoạn ảnh màu HSV / Canny
│
├── calibration/                    # 🧪 CÂN CHỈNH PHẦN CỨNG & THỊ GIÁC
│   ├── calib_bev.py                #   Cân chỉnh ma trận phối cảnh BEV
│   ├── calib_hsv.py                #   Cân chỉnh dải màu HSV vạch đường
│   ├── calib_speed.py              #   Cân chỉnh tốc độ & góc cua tròn 90°
│   └── test_camera_csi.py          #   Kiểm tra trực tiếp camera CSI
│
├── setup/                          # 🛠️ SCRIPTS THIẾT LẬP HỆ THỐNG
│   ├── setup_hardware.sh           #   Cài đặt driver & thư viện xe
│   ├── launch_camera.sh            #   Khởi chạy camera ROS node
│   ├── launch_lidar.sh             #   Khởi chạy LiDAR node
│   └── launch_joy.sh               #   Khởi chạy Gamepad Joy node
│
├── tests/                          # 🧪 BỘ KIỂM THỬ TỰ ĐỘNG (Automated Unit Tests)
│   ├── test_dagger_pipeline.py     #   Kiểm thử toàn diện hệ thống DAgger 15D
│   ├── test_onnx_inference.py      #   Kiểm thử suy luận mô hình ONNX
│   ├── test_import_model.py        #   Kiểm thử TensorRT / CUDA / ONNXRuntime
│   ├── test_lane_detection.py      #   Kiểm thử phát hiện vạch làn V3
│   └── test_pid_controller.py      #   Kiểm thử bộ điều khiển PID
│
├── models/                         # 📦 MÔ HÌNH TRỌNG SỐ (ONNX / Weights)
├── logs/                           # 📊 DỮ LIỆU LOGS & SESSIONS TELEMETRY
├── catkin_ws/                      # 🤖 ROS Melodic Workspace
├── archive/                        # 📦 Code cũ dự phòng từ Hackathon
└── README.md                       # 📖 Hướng dẫn tổng quan này
```

---

## 🏁 Hướng Dẫn Chạy Các Bài Thi

### 1. Bài 1: Speed Track (30% điểm)
* **File thực thi:** `runners/speed_track/main_speed_track_v3.py`
* **Thuật toán:** Thị giác BEV Multi-Lane RANSAC V3 + Cảm biến LiDAR phát hiện vật cản + Bộ điều khiển tối ưu **LQR Controller với S-Curve Offset Ramping**.
* **Lệnh chạy:**
  ```bash
  python3 runners/speed_track/main_speed_track_v3.py
  ```

### 2. Bài 2: Smart City (40% điểm)
* **File thực thi:** `runners/smart_city/main_smart_city.py`
* **Thuật toán:** Tìm đường ngắn nhất đồ thị A*/Dijkstra ([`robot/planning/map_navigator.py`](robot/planning/map_navigator.py)) + Máy trạng thái giao lộ FSM + Nhận diện biển báo & đèn giao thông YOLO / Roboflow + Quét mã QR (`pyzbar`) + MQTT.
* **Lệnh chạy:**
  ```bash
  python3 runners/smart_city/main_smart_city.py
  ```

### 3. Học Bắt Chước Trực Tuyến & Né Vật Cản (DAgger 15D)
* **File thực thi:** `runners/dagger/main_dagger.py`
* **Thuật toán:** Vector trạng thái $S_t \in \mathbb{R}^{15}$ (bổ sung độ cong $\kappa$, tốc độ trôi ngang $\dot{e}_y$, chênh lệch khoảng trống 2 sườn $\Delta d_{\text{side}}$, lịch sử góc lái $a_{t-1}$) + Mạng Dual-Head MLP có LayerNorm + Bộ đệm phân tầng Stratified Replay Buffer + Lớp an toàn LiDAR Safety Layer.
* **Lệnh chạy:**
  ```bash
  python3 runners/dagger/main_dagger.py
  ```
  * **Triangle (Button 3):** Mở khóa cho xe chạy.
  * **Cần Joy Left Stick:** Nhích cần để can thiệp lái mẫu khi gặp khúc cua khó hoặc vật cản (AI tự động ghi nhận và fine-tune trực tuyến).
  * **L1 (Button 4):** Lưu checkpoint model ngay lập tức.
  * **Circle (Button 1):** Phanh khẩn cấp (E-STOP).

* **Huấn luyện Offline mồi mô hình:**
  ```bash
  python3 training/train_dagger_offline.py --epochs 200 --batch-size 64
  ```

---

## 🚀 Quy Trình Vận Hành Trên Xe Thật

### Bước 1: Khởi động ROS Core & Cảm Biến
* **Terminal 1:**
  ```bash
  roscore
  ```
* **Terminal 2 (Khởi chạy Camera & LiDAR):**
  ```bash
  source /opt/ros/melodic/setup.bash
  source ~/catkin_ws/devel/setup.bash
  roslaunch jetracer lidar.launch
  roslaunch jetracer csi_camera.launch
  ```

### Bước 2: Kích hoạt Môi Trường & Chạy Thuật Toán
* **Terminal 3:**
  ```bash
  source ~/my_env/bin/activate
  cd ~/robot-jeston
  # Chạy bài thi mong muốn:
  python3 runners/speed_track/main_speed_track_v3.py
  ```

---

## 🧪 Kiểm Thử Tự Động (Automated Testing)

Chạy bộ test suite kiểm tra toàn diện hệ thống:
```bash
python3 tests/test_dagger_pipeline.py
python3 tests/test_onnx_inference.py
python3 tests/test_import_model.py
```

---
*Chúc đội thi **PromptEngineer** thi đấu tự tin và đạt thành tích cao nhất! 🚀🏆*