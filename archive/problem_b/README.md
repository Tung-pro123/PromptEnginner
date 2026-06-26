# JetBot Event-Driven Controller 🚗⚡

Hệ thống điều khiển JetBot dựa trên **sự kiện (event-driven)** chạy trên **ROS**, bám vạch bằng **OpenCV**, phát hiện giao lộ bằng **LiDAR**, điều hướng theo **bản đồ đồ thị**, và (tuỳ chọn) nhận diện biển báo qua **Roboflow Detection API**. Mã nguồn chính nằm trong `ros_lidar_follower.py`. (Tham chiếu nội dung file gốc có trong repo.) 

> Mục tiêu: đi từ **start_node** đến **end_node** theo **đường đi tối ưu**, an toàn và ổn định.

---

## 🔧 Thành phần chính

- **FSM (Finite State Machine)**: `WAITING_FOR_LINE → DRIVING_STRAIGHT → APPROACHING_INTERSECTION → HANDLING_EVENT → LEAVING_INTERSECTION → REACQUIRING_LINE → (GOAL_REACHED | DEAD_END)`
- **Bám vạch (OpenCV)**: lọc theo HSV trong 2 ROI (chính + dự báo) để giữ hướng ổn định.
- **LiDAR Intersection Detector**: phát hiện giao lộ đáng tin cậy, ưu tiên cao hơn thị giác.
- **Điều hướng theo bản đồ**: đọc `map.json`, tìm đường (hỗ trợ `find_shortest_path_through_loads` nếu có).
- **Nhận diện (tuỳ chọn)**: dùng **Roboflow** thay cho YOLO ONNX cục bộ; trả kết quả đồng nhất với pipeline cũ.
- **MQTT Publisher**: đẩy dữ liệu “event/data” ra topic để dashboard/giám sát.
- **Ghi video debug**: `jetbot_run.avi` với overlay ROI + trạng thái.

> Mã nguồn triển khai chi tiết các luồng trên, bao gồm khởi tạo phần cứng JetBot (hoặc mock), Roboflow API, MQTT, ghi hình, và logic xử lý tại giao lộ. :contentReference[oaicite:0]{index=0}

---

## 🗺️ Kiến trúc topic ROS

- Camera: `/csi_cam_0/image_raw`
- LiDAR: `/scan`
- Node chính: `jetbot_controller_node`

---

## 📁 Cấu trúc khuyến nghị


## Hướng dẫn chạy nhanh

Hãy kết nối vào jetbot, sau đó thực hiện lần lượt 3 câu lệnh sau trong terminal:

```sh
roslaunch jetbot_pro lidar.launch
roslaunch jetbot_pro csi_camera.launch
python3 ros_lidar_follower.py
```
---
