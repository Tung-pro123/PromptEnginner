# 📘 Báo Cáo Review Mã Nguồn - Nhánh `quyen`

Tài liệu này tổng hợp phân tích chi tiết mã nguồn, kiến trúc hệ thống và giải pháp kỹ thuật trên nhánh **`quyen`**, đồng thời so sánh với phương án của nhánh **`tung`**.

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG
Nhánh `quyen` tập trung hoàn thành bài toán **Smart City (Bài toán Sa bàn Đô thị Thông minh - Bài A, B, C)**. Kiến trúc được xây dựng theo mô hình **Event-Driven & Map-Based Navigation**:

```
[ map.json ] ──> [ MapNavigator ] ──> [ JetBotController (FSM 8 Trạng Thái) ]
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
             [ Camera / ONNX ]       [ LiDAR / Opposite ]     [ MQTT / QR Code ]
             (Lane & Signs)          (Intersections)          (Server Scoring)
```

### Các tập tin & mô-đun chính:
* **`src/smart_city/main_smart_city.py`**: Chương trình điều khiển trung tâm (870+ dòng code) tích hợp toàn bộ luồng hoạt động Đô thị Thông minh.
* **`src/core/planning/map_navigator.py`**: Mô-đun xử lý đồ thị bản đồ, lập kế hoạch lộ trình ngắn nhất qua các điểm mốc.
* **`src/core/utils/opposite_detector.py`**: Phát hiện chướng ngại vật / xe đi ngược chiều.
* **Tích hợp phần cứng & giao tiếp:** ONNX Runtime (AI nhận diện biển báo), Pyzbar (Đọc QR Code), Paho-MQTT (Gửi dữ liệu chấm điểm về Server).

---

## 2. CHI TIẾT CÁC MÔ-ĐUN KỸ THUẬT NỔI BẬT

### 🧠 A. Máy trạng thái 8 bước (FSM 8 States)
Hệ thống điều khiển trung tâm quản lý 8 trạng thái hoạt động chặt chẽ:
1. `WAITING_FOR_LINE`: Chờ phát hiện vạch đường tại điểm xuất phát.
2. `DRIVING_STRAIGHT`: Bám làn di chuyển thẳng giữa hai nút giao.
3. `APPROACHING_INTERSECTION`: Giảm tốc độ và phát hiện ngã tư/giao lộ (ROI phía xa + LiDAR).
4. `HANDLING_EVENT`: Dừng tại giao lộ, tra bản đồ, đọc biển báo/đèn giao thông/mã QR.
5. `LEAVING_INTERSECTION`: Thực thi lệnh rẽ ($90^\circ$ trái/phải hoặc đi thẳng) để ra khỏi ngã tư.
6. `REACQUIRING_LINE`: Khôi phục lại góc nhìn và bám lại làn đường mới.
7. `DEAD_END`: Xử lý khi gặp đường cấm/đường đán (kích hoạt Re-planning).
8. `GOAL_REACHED`: Hoàn thành lộ trình và dừng xe an toàn tại điểm đích.

### 🗺️ B. Lập kế hoạch Đường đi Tối ưu qua nhiều Điểm mốc (`MapNavigator`)
* **Đọc bản đồ `map.json`:** Khởi tạo đồ thị có hướng (Directed Graph) biểu diễn các nút giao và khoảng cách giữa các node.
* **Thuật toán `find_shortest_path_through_loads`:**
  * Tự động tính toán chuỗi lộ trình ngắn nhất đi qua lần lượt từng điểm nhận/trả hàng: `start` $\rightarrow$ `load_1` $\rightarrow$ `load_2` $\rightarrow$ ... $\rightarrow$ `end`.
* **Cấm đường động (Dynamic Re-planning):**
  * Khi phát hiện đường tắc hoặc vật cản ngược chiều, hệ thống tự động thêm cạnh bị lỗi vào danh sách `banned_edges` và tính toán lại lộ trình tức thì từ vị trí hiện tại.

### 📷 C. Nhận diện Biển báo AI (ONNX) & Đọc Mã QR (Pyzbar)
* **ONNX Runtime (`initialize_yolo`):** Sử dụng mô hình AI nhẹ nén dạng ONNX để nhận diện nhanh các loại biển báo giao thông trên Jetson Nano.
* **Giải mã QR Code (`pyzbar`):** Tự động quét và đọc dữ liệu mã QR tại các điểm dừng nhận/trả hàng để xác nhận hoàn thành nhiệm vụ.
* **Gửi dữ liệu chấm điểm (MQTT):** Giao tiếp thời gian thực với Server Ban tổ chức qua cổng Paho-MQTT (`ROS_TOPIC_JOY`, `status`, `node_id`).

---

## 3. SO SÁNH GIỮA NHÁNH `quyen` VÀ NHÁNH `tung` (BẠN)

| Tiêu chí | Nhánh của **TÙNG** (`tung`) | Nhánh của **QUYỀN** (`quyen`) |
| :--- | :--- | :--- |
| **Phạm vi Bài toán** | **Speed Track (Bài 1):** Đua tốc độ, bám làn và né vật cản động không sử dụng bản đồ. | **Smart City (Bài A, B, C):** Sa bàn đô thị thông minh có ngã tư, bản đồ `map.json`, biển báo và mã QR. |
| **Xử lý Ảnh Bám làn** | **Dual-Filter (HSV Red + White Background):** Lọc vạch đỏ và nền trắng xung quanh đường đen cực kỳ chính xác cho xa hình mới. | **Single ROI / Thresholding:** Tập trung bám làn thẳng và nhận diện vạch cắt ngang tại các ngã tư. |
| **Né vật cản & Trả làn** | **State-Aware Segment Clustering:** Né vật cản 2 phía bằng LiDAR, có Safety Override góc lái $\ge 0.28$ và trả làn 2 giai đoạn. | **Opposite Detector & Re-planning:** Phát hiện vật cản ngược chiều/tắc đường và kích hoạt quay đầu hoặc cấm đường (`banned_edges`). |
| **Tích hợp Server & AI** | Tập trung xử lý offline điều khiển xe nhanh nhất. | **Tích hợp đầy đủ ONNX AI Model, đọc QR Code Pyzbar và gửi dữ liệu chấm điểm Server qua MQTT.** |

---

## 4. ĐÁNH GIÁ CHUNG & ĐỀ XUẤT TÍCH HỢP

* **Ưu điểm lớn nhất của Quyền:** Đã hoàn thiện toàn bộ **hạ tầng phần mềm cho Bài toán Đô thị Thông minh (Bài A/B/C)** bao gồm đọc bản đồ, điều hướng ngã tư, đọc mã QR và gửi dữ liệu chấm điểm MQTT.
* **Điểm kết hợp hoàn hảo:**
  * Dùng **`main_speed_track.py` của Tùng** cho phần đua tốc độ Bài 1 (Speed Track).
  * Dùng **`main_smart_city.py` của Quyền** làm bộ khung chính cho các bài thiSa bàn Đô thị.
  * Đưa thuật toán xử lý màu kép & chống nhiễu lệch biên của Tùng vào module `lane_detector.py` của Quyền để giúp xe đi qua các khúc cua ngã tư mượt mà nhất.
