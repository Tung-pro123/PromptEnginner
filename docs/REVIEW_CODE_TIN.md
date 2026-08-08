# 📘 Báo Cáo Review Mã Nguồn - Nhánh `tin` (Tín)

Tài liệu này tổng hợp phân tích chi tiết mã nguồn, kiến trúc AI TensorRT và giải pháp kỹ thuật trên nhánh **`tin`** (Tín), đồng thời so sánh với phương án của nhánh **`tung`**.

---

## 1. TỔNG QUAN GIẢI PHÁP KỸ THUẬT
Nhánh `tin` tập trung xây dựng **Hệ thống AI Nhận diện Biển báo & Đèn giao thông tối ưu bằng YOLOv5n TensorRT** và thiết lập **Kiến trúc ROS Package chuẩn chuyên nghiệp (`jetracer_smartcity`)** cho Bài toán Đô thị Thông minh.

```
 [ Camera Node (/image_raw) ]
              │
              ▼
 [ AI Bridge Node (Python 3) ] ──> [ YoloDetector (TensorRT FP16 Engine) ]
              │                                      │
              ▼                                      ▼
    [ Custom Msg /detections ] ◄────────── [ Fallback HSV Color Classifier ]
              │
              ▼
 [ Intersection State Machine ] ──> [ Control Node ] ──> [ Motor Driver (/cmd_vel) ]
```

---

## 2. CHI TIẾT CÁC MÔ-ĐUN KỸ THUẬT NỔI BẬT

### 🤖 A. Pipeline Huấn luyện & Tối ưu AI Engine (`training/` & `best.onnx`)
1. **Huấn luyện YOLOv5 Nano (`train_yolov5n.ipynb`):**
   * Huấn luyện mô hình siêu nhẹ YOLOv5n chuyên biệt cho bài toán nhận diện biển báo và đèn giao thông trên Jetson Nano.
2. **Xuất mô hình ONNX & TensorRT FP16 (`export_onnx.py` & `export_tensorrt.sh`):**
   * Chuyển đổi mô hình PyTorch $\rightarrow$ ONNX (`opset 11`) $\rightarrow$ **TensorRT `.engine` với độ chính xác bán nguyên FP16 (Half-precision)** trực tiếp trên Jetson Nano.
   * **Hiệu năng:** Đạt tốc độ suy luận (Inference Speed) vượt trội **25 - 30 FPS** trên GPU CUDA của Jetson Nano mà không gây ngốn CPU.

### 🌉 B. Cầu nối ROS Python 3 Bridge (`ai_bridge_node.py` & `jetracer_py3_bridge`)
* **Giải quyết xung đột Python 2 / Python 3:**
  * ROS Melodic mặc định dùng Python 2.7, trong khi TensorRT / PyTorch / OpenCV hiện đại yêu cầu Python 3.
  * Tín đã viết `ai_bridge_node.py` chạy riêng ở Python 3, subscribe topic `/image_raw`, thực thi TensorRT inference và xuất dữ liệu vật thể phát hiện được lên ROS topic `/detections` dưới định dạng tin nhắn tùy chỉnh `DetectionArray.msg`.

### 🛡️ C. Bảo hiểm 2 lớp Lọc Đèn giao thông (`traffic_light_state.py`)
* **Lớp bảo hiểm HSV:** Nếu YOLOv5n phát hiện được khung hình đèn giao thông nhưng độ tin cậy (Confidence) bị sụt giảm do ánh sáng phòng lab thay đổi, module `traffic_light_state.py` tự động cắt vùng ảnh (Crop BBox) và dùng bộ lọc màu HSV (Đỏ / Xanh) để kiểm chứng lại 100% màu đèn thật.

### 🗺️ D. Điều hướng Ngã tư & Ghi Log Chấm điểm (`route_planner.py` & `run_logger.py`)
* **Route Planner:** Parse cấu trúc đồ thị từ file `intersection_map.yaml`, chạy thuật toán BFS/Dijkstra tìm đường ngắn nhất giữa các ngã tư.
* **Run Logger:** Tự động ghi vết log hành trình chạy ra file CSV theo đúng chuẩn sơ đồ chấm điểm yêu cầu của Ban tổ chức.

---

## 3. SO SÁNH GIỮA NHÁNH `tin` VÀ NHÁNH `tung` (BẠN)

| Tiêu chí | Nhánh của **TÙNG** (`tung`) | Nhánh của **TÍN** (`tin`) |
| :--- | :--- | :--- |
| **Phạm vi Bài toán** | **Speed Track (Bài 1):** Đua tốc độ bám làn & né vật cản động không sử dụng bản đồ (Mapless). | **Smart City (Bài 2):** Nhận diện biển báo, đèn giao thông bằng AI YOLOv5n TensorRT & điều hướng ngã tư. |
| **Phương pháp Xử lý Ảnh** | **OpenCV Hình học (Dual-Filter HSV Red + White Background):** Nhẹ, chạy trực tiếp trên CPU không cần mô hình AI. | **Deep Learning AI (YOLOv5n + TensorRT FP16 Engine):** Tận dụng sức mạnh phần cứng GPU CUDA của Jetson Nano. |
| **Cấu trúc Gói ROS** | Code tập trung trong file chính để chạy trực tiếp và debug dễ dàng. | Đóng gói thành **ROS Package chuẩn (`jetracer_smartcity`)** với đầy đủ file `launch`, `msg` tùy chỉnh, `config` YAML. |
| **Tài liệu & Pipeline** | Hướng dẫn chạy file và nhật ký chẩn đoán lượt chạy thực tế. | Thiết lập pipeline huấn luyện từ Colab $\rightarrow$ ONNX $\rightarrow$ TensorRT và sơ đồ kiến trúc Mermaid chi tiết (`smartcity_architecture.md`). |

---

## 4. ĐÁNH GIÁ CHUNG & TỔNG KẾT TOÀN ĐỘI

* **Ưu điểm lớn nhất của Tín:** Tín đã giải quyết xuất sắc bài toán **tăng tốc AI trên phần cứng Jetson Nano bằng TensorRT FP16**, giúp mô hình nhận diện biển báo/đèn giao thông chạy cực kỳ mượt mà ở tốc độ frame cao. Đồng thời cấu trúc ROS Package rất chuẩn mực.

---

# 🏆 TỔNG HỢP VAI TRÒ & SỰ PHỐI HỢP CỦA CÁC THÀNH VIÊN

1. **TÙNG (`tung`):** Chịu trách nhiệm chính **Thuật toán Speed Track (Bài 1)** - Bám làn màu kép, gán nhãn biên FSM State-Aware, né vật cản động LiDAR và trả làn 2 giai đoạn (đã test thành công trên xe thực tế).
2. **LÂM (`lam`):** Chịu trách nhiệm **Khung Kiến trúc Dự án (Layered Architecture & Blackboard Pattern)** - Mô-đun hóa dự án sạch sẽ, quản lý tham số `settings.py` và viết bộ điều khiển PID có Anti-windup.
3. **QUYỀN (`quyen`):** Chịu trách nhiệm **Máy Trạng Thái Sa Bàn Đô Thị (Smart City FSM 8 Steps & MapNavigator)** - Quản lý luồng chuyển giao lộ, thuật toán tìm đường qua nhiều mốc `map.json`, đọc mã QR `pyzbar` và kết nối Server MQTT.
4. **NHẤT (`hnyat`):** Chịu trách nhiệm **Toán học Điều khiển Tối ưu (LQR Controller)** - Mô hình xe đạp động học, giải phương trình DARE Riccati để dập tắt dao động bẻ lái và xây dựng tập hợp script test phần cứng `tests/`.
5. **TÍN (`tin`):** Chịu trách nhiệm **Hệ thống AI & ROS Package (YOLOv5n TensorRT & ROS Py3 Bridge)** - Train model AI, convert TensorRT `.engine` chạy 30 FPS trên GPU và đóng gói gói ROS `jetracer_smartcity`.

> 💡 **Kết luận:** Cả 5 thành viên trong đội đều đang làm việc đúng hướng và bổ trợ hoàn hảo cho nhau! Đội của bạn đang có một lực lượng mã nguồn cực kỳ mạnh mẽ để tự tin chinh phục cả Bài 1 (Speed Track) và Bài 2/3 (Smart City)!
