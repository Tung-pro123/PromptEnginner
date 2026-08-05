# JetRacer Smart City System - Jetson AI Racer Challenge 2026

Dự án này triển khai hệ thống tự hành cho xe NVIDIA JetRacer tham gia cuộc thi Jetson AI Racer Challenge 2026 (Bài 2: Smart City), chạy trên nền tảng ROS Melodic, Jetpack 4.5.1, CUDA 10.2 và mô hình YOLOv5n TensorRT FP16.

---

## 1. Hướng dẫn thiết lập môi trường trên Jetson Nano

Do Jetson Nano chạy Jetpack 4.5.1 dùng Ubuntu 18.04 có cấu hình phần cứng hạn chế và phân tách Python 2 (mặc định của ROS Melodic) và Python 3 (cho AI), hãy làm theo hướng dẫn tối ưu bên dưới:

### Bước 1: Cài đặt các thư viện cần thiết cho Python 3
Mở Terminal trên Jetson Nano và cài đặt các dependency cho xử lý ảnh, AI và truyền nhận thông tin:

```bash
# Cập nhật hệ thống
sudo apt-get update
sudo apt-get install -y python3-pip python3-matplotlib python3-numpy python3-scipy python3-opencv

# Cài đặt ZeroMQ cho giao tiếp liên tiến trình (nếu dùng cầu nối Python 2 - Python 3)
pip3 install pyzmq

# Cài đặt PyCUDA để giao tiếp với GPU thông qua Python 3
# Lưu ý: Cần thêm CUDA vào PATH trước khi cài pycuda
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
pip3 install pycuda
```

### Bước 2: Cài đặt PyTorch tương thích GPU CUDA 10.2
Tải bản build sẵn (`.whl`) chính thức từ NVIDIA cho Jetpack 4.5:
```bash
wget https://nvidia.box.com/shared/static/p57jw14s46hy43hzcg9g3317k27956pj.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```

### Bước 3: Cài đặt thư viện TensorRT cho Python 3
Thông thường TensorRT đã đi kèm sẵn trong Jetpack. Bạn cần kiểm tra xem thư viện python của TensorRT đã liên kết chính xác chưa bằng lệnh:
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```
Nếu bị lỗi không tìm thấy module `tensorrt`, hãy copy hoặc tạo symlink liên kết thư viện từ hệ thống vào thư mục Python 3 site-packages của bạn.

---

## 2. Biên dịch Workspace ROS

Workspace được tổ chức tại thư mục `catkin_ws/`. Để biên dịch hệ thống và các custom messages:

```bash
# Di chuyển đến thư mục catkin_ws
cd d:/AI_Project/racing_promax/catkin_ws

# Biên dịch các package
catkin_make

# Nạp biến môi trường cho phiên làm việc hiện tại
source devel/setup.bash
```

---

## 3. Quy trình chạy thực tế trên xe

Hệ thống hỗ trợ cả 2 chế độ tùy thuộc vào môi trường trên xe:

### Chế độ A: Chạy Native Python 3 (Khuyên Dùng nếu ROS hỗ trợ Python 3)
Nếu môi trường Python 3 của bạn import được `rospy` và `cv_bridge`, khởi chạy trực tiếp bằng:
```bash
roslaunch jetracer_smartcity smartcity.launch
```

### Chế độ B: Chạy qua ZeroMQ Bridge (Phương án dự phòng cực kỳ ổn định)
Nếu Python 3 không thể tương tác trực tiếp với ROS, hãy chạy thông qua Bridge giao tiếp giữa Python 2 và Python 3:

1. **Khởi chạy phần ROS điều khiển và Camera (Python 2)**:
   ```bash
   # Terminal 1: Chạy camera driver và các node điều khiển động cơ của xe
   source devel/setup.bash
   roslaunch jetracer_smartcity smartcity.launch use_bridge:=true
   ```

2. **Khởi chạy phần xử lý AI (Python 3)**:
   ```bash
   # Terminal 2: Chạy pipeline xử lý AI độc lập
   cd d:/AI_Project/racing_promax/catkin_ws/src/jetracer_smartcity/scripts
   python3 ai_bridge_node.py --use_zmq
   ```

---

## 4. Biên dịch mô hình sang TensorRT (.engine)

Để tối ưu hóa FPS >= 20 và độ trễ <= 300ms, ta chuyển đổi mô hình từ PyTorch sang TensorRT ngay trên xe Jetson:

1. **Xuất mô hình ra định dạng ONNX (trên máy cá nhân/Colab)**:
   ```bash
   python export_onnx.py --weights yolov5n_smartcity.pt
   ```
2. **Convert ONNX sang Engine TensorRT FP16 (trên xe Jetson Nano)**:
   ```bash
   trtexec --onnx=yolov5n_smartcity.onnx --saveEngine=yolov5n_smartcity.engine --fp16 --workspace=1024
   ```
3. Copy file `.engine` thu được vào thư mục `models/` của project.

---

## 5. Xem Log thi đấu
Kết quả chạy của xe sẽ tự động được ghi nhận tại thư mục `logs/` dạng file CSV với schema chuẩn từ BTC nhằm phục vụ phân tích độ trễ và giải quyết tranh chấp.
