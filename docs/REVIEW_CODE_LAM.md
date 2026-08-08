# 📘 Báo Cáo Review Mã Nguồn - Nhánh `lam`

Tài liệu này tổng hợp phân tích chi tiết mã nguồn, kiến trúc hệ thống và giải pháp kỹ thuật trên nhánh **`lam`**, đồng thời so sánh với phương án của nhánh **`tung`**.

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG
Nhánh `lam` được thiết kế theo **Kiến trúc Phân lớp (Layered Architecture)** kết hợp với mô hình **Bảng bộ nhớ chung (Blackboard Pattern)**:

```
[ Perception Layer ] ---> [ Blackboard ] ---> [ AI & FSM Layer ] ---> [ Control Layer ]
  (Camera/LiDAR/Traffic)   (State Memory)     (Decision Engine)      (PID/Predictive)
```

### Các phân lớp chính:
* **`src/config/settings.py`**: Quản lý tập trung toàn bộ tham số cấu hình hệ thống (Camera, LiDAR, PID, AI, Traffic).
* **`src/perception/`**: Tầng cảm biến & thị giác máy tính (Camera, LiDAR, Đèn giao thông, Biển báo).
* **`src/fsm/` & `src/ai/`**: Tầng điều phối máy trạng thái né vật cản & AI ra quyết định hành vi cấp cao (Priority Chain).
* **`src/control/`**: Tầng điều khiển ga/lái phần cứng (PID Controller, Predictive Controller, Motor Driver).

---

## 2. CHI TIẾT CÁC MÔ-ĐUN KỸ THUẬT NỔI BẬT

### 📷 A. Xử lý Ảnh & Bám làn (`src/perception/camera/camera_processor.py`)
1. **Lọc dải màu HSV Cam/Đỏ:**
   * Dải 1 (Đỏ nhạt / Cam): $H \in [0, 22]$, $S \in [60, 255]$, $V \in [50, 255]$.
   * Dải 2 (Đỏ đậm): $H \in [160, 180]$, $S \in [60, 255]$, $V \in [50, 255]$.
2. **Khép khuyết đứt gãy (Morphology Closing):**
   * Sử dụng `cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)` với kernel $5 \times 5$ để tự động nối liền các vết nứt đứt đoạn trên vạch kẻ đường.
3. **Bộ lọc làm mượt EMA (Exponential Moving Average):**
   * Sử dụng hệ số $\alpha = 0.6$ làm mượt tọa độ Waypoints tâm đường giữa các khung hình liên tiếp, giảm thiểu hiện tượng bánh lái bị giật đột ngột.
4. **Cổng chờ Phân đoạn ảnh AI:**
   * Tích hợp sẵn biến `USE_ADVANCED_SEGMENTATION` kết nối tới `detect_lane.py` cho các bài toán phân đoạn ảnh bằng Deep Learning trong tương lai.

### 🚦 B. Bộ não AI & Nhận diện Đèn/Biển báo (`src/perception/camera/traffic_detector.py` & `src/ai/ai_decision_engine.py`)
1. **Traffic Detector (Đèn & Biển báo):**
   * **Vùng quét ROI:** Chỉ tập trung quét $45\%$ chiều cao phía trên ảnh (`TRAFFIC_ROI_TOP_RATIO = 0.45`) để tránh nhiễu với lòng đường.
   * **Thuật toán Bầu chọn (Voting):** Theo dõi lịch sử trong $5$ khung hình liên tiếp (`TRAFFIC_HISTORY_LEN = 5`) để loại bỏ hoàn toàn các khung hình nhận dạng sai (false positive).
2. **AI Decision Engine (Chuỗi ưu tiên hành vi - Priority Chain):**
   * `WAIT_RED_LIGHT`: Ưu tiên số 1 - Phanh dừng xe khi phát hiện đèn đỏ.
   * `TURN_LEFT` / `TURN_RIGHT` / `GO_STRAIGHT`: Ưu tiên số 2 - Khi mất vạch liên tiếp $5$ frames (xác nhận ngã tư), điều khiển xe rẽ hoặc đi thẳng theo đọc biển báo chỉ dẫn.
   * `FOLLOW_LANE`: Ưu tiên số 3 - Chế độ mặc định bám làn đường.

### 🎮 C. Điều khiển Động cơ (`src/control/pid_controller.py`)
* **Bộ điều khiển PID hoàn chỉnh:**
  * **P (Proportional):** Tỷ lệ với sai số lệch tâm đường.
  * **I (Integral) + Anti-Windup:** Tích phần sai số có giới hạn `max(-1.0, min(1.0, integral))` tránh trôi góc lái.
  * **D (Derivative):** Tốc độ thay đổi sai số để giảm văng đuôi.
  * **Vùng an toàn (Dead Zone):** Sai số $< 3\%$ chiều rộng ảnh sẽ giữ nguyên góc lái thẳng.
* **Chế độ Mô phỏng (Mock Mode):** Tự động chuyển sang Mock Class nếu không phát hiện phần cứng JetRacer thật, giúp chạy code thử nghiệm trên máy tính cá nhân dễ dàng.

---

## 3. SO SÁNH GIỮA NHÁNH `lam` VÀ NHÁNH `tung` (BẠN)

| Tiêu chí | Nhánh của **TÙNG** (`tung`) | Nhánh của **LÂM** (`lam`) |
| :--- | :--- | :--- |
| **Cấu trúc Mã nguồn** | Tập trung trực tiếp trong `main_speed_track.py` để đạt tốc độ thực thi tối đa, dễ debug 1 file và chỉnh thông số ngay trên sa bàn. | Mô-đun hóa phân lớp (`perception`, `ai`, `control`, `fsm`), code sạch, tuân thủ chuẩn thiết kế OOP & Blackboard Pattern. |
| **Thuật toán Bám làn** | **Dual-Filter (HSV Red + White Background):** Lọc cả vạch biên đỏ và nền trắng xung quanh đường đen. | **Single-Filter (HSV Orange/Red):** Lọc dải màu cam/đỏ kết hợp Morphology Closing. |
| **Chống nhiễu lệch biên** | **State-Aware Segment Clustering:** Gom cụm và gán nhãn biên theo FSM/Zone-based, triệt tiêu lỗi nhận nhầm vạch khi xe nghiêng sâu. | **Làm mượt EMA ($\alpha = 0.6$):** Làm mượt tọa độ waypoints tâm đường theo thời gian. |
| **Né vật cản & Trả làn** | **Safety Override** (khóa góc lái $\ge 0.28$) và **Giao thức trả làn 2 giai đoạn** (ép lái mở $0.50$ trong 1.2s). | Dùng cơ chế Ramping Offset tiêu chuẩn ($5.0$px/frame). |
| **Đèn & Biển báo** | Chưa tích hợp (tập trung tối đa cho Speed Track). | **Đã có sẵn module `traffic_detector` & `ai_decision_engine`** để xử lý ngã tư và đèn giao thông. |

---

## 4. ĐÁNH GIÁ CHUNG & ĐỀ XUẤT TÍCH HỢP

* **Ưu điểm lớn nhất của Lâm:** Khung kiến trúc code được tổ chức **cực kỳ chuyên nghiệp, dễ mở rộng**. Đã dựng sẵn bộ não AI xử lý ngã tư và đèn giao thông.
* **Điểm cần bổ sung cho Lâm:** Thuật toán xử lý ảnh bám làn & né vật cản thực tế trên sa bàn chưa có các cơ chế chống nhiễu chuyên sâu (như gán nhãn biên theo FSM hay ép góc lái trả làn mở).
* **Khuyên dùng khi hợp nhất:** Giữ nguyên khung cấu trúc mô-đun của Lâm làm khung dự án chung, sau đó lấy core xử lý ảnh & né vật cản từ `main_speed_track.py` của Tùng đưa vào `camera_processor.py` và `fsm_manager.py` của Lâm.
