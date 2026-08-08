# 📋 DẢY TIẾN ĐỘ & BẢNG CÔNG VIỆC (TODO-LIST) DỰ ÁN JETSON AI RACER 2026
**Đội thi:** PromptEngineer  
**Cập nhật ngày:** 08/08/2026  

---

## 📌 1. TỔNG QUAN TIẾN ĐỘ HIỆN TẠI (CURRENT STATUS)

- [x] **Bài 1: Speed Track (Tùng):** 
  - Đã hoàn thành thuật toán bám làn màu kép (HSV Red + White Background) cho xa hình mới.
  - Đã kiểm thử thành công né vật cản 2 phía bằng LiDAR, chống nhiễu lệch biên theo trạng thái FSM (State-Aware Segment Clustering) và trả làn 2 giai đoạn trên xe thật.
- [x] **Khung Kiến trúc Phân lớp (Lâm):** 
  - Đã hoàn thành tái cấu trúc dự án theo Layered Architecture + Blackboard Pattern và file cấu hình tập trung `settings.py`.
- [x] **Sa bàn Đô thị Smart City (Quyền):** 
  - Đã xây dựng máy trạng thái FSM 8 bước, lập kế hoạch đường đi `MapNavigator` qua nhiều node `map.json`, đọc mã QR và giao tiếp MQTT Server.
- [x] **Bộ điều khiển LQR & Test Suite (Nhất):** 
  - Đã giải xong phương trình Riccati (DARE) cho bộ điều khiển tối ưu LQR và xây dựng bộ script test phần cứng độc lập `tests/`.
- [x] **AI YOLOv5n TensorRT Engine (Tín):** 
  - Đã train xong model YOLOv5n, convert TensorRT FP16 chạy 30 FPS trên Jetson GPU và dựng ROS Python 3 Bridge node.

---

## 💡 2. ĐỀ XUẤT CẢI TIẾN ĐẮC GIÁ (HIGH-VALUE IMPROVEMENTS)

### 🔥 Đề xuất 1: Nâng cấp Bộ điều khiển P $\rightarrow$ LQR cho Bài Speed Track (Tùng + Nhất)
* **Ý tưởng:** Kết hợp thuật toán bám làn Dual-Filter & FSM Né vật cản của **Tùng** với Bộ điều khiển Tối ưu **LQR** (`lqr_controller.py`) của **Nhất**.
* **Hiệu quả:** Triệt tiêu hoàn toàn hiện tượng văng đuôi hay lắc bánh khi vào cua gấp hoặc lách né hộp carton. Xe lượn cua theo đường cong tiếp tuyến cực kỳ mượt mà.

### 🔥 Đề xuất 2: Tối ưu AI Nhận diện Biển báo bằng TensorRT FP16 (Tín + Quyền)
* **Ý tưởng:** Ghép mô hình TensorRT `.engine` chạy 30 FPS của **Tín** vào module nhận diện ngã tư `main_smart_city.py` của **Quyền**.
* **Hiệu quả:** Xe nhận biết biển báo và đèn giao thông ngay từ khoảng cách 1.5m với độ trễ gần như bằng 0, giúp xe chuẩn bị tinh thần rẽ từ xa mà không phải phanh gấp tại ngã tư.

### 🔥 Đề xuất 3: Chuẩn hóa Tập trung Cấu hình `settings.py` (Lâm + Cả nhóm)
* **Ý tưởng:** Đưa toàn bộ các tham số thực tế từ xe (như `DODGE_OFFSET_PX`, `TRIGGER_DIST`, `RAMP_STEP_PX`, `PID_KP`, `AI_SPEED`) vào duy nhất 1 file `src/config/settings.py`.
* **Hiệu quả:** Khi ra sa bàn thi đấu, cả đội chỉ cần chỉnh tham số ở 1 vị trí duy nhất mà không cần can thiệp sâu vào code thuật toán.

### 🔥 Đề xuất 4: Launcher Khởi chạy 1 Lệnh Duy Nhất (`production.launch`)
* **Ý tưởng:** Gộp cả `sensors.launch`, `ai_bridge_node`, `nvargus-daemon restart` và script chạy chính thành 1 file ROS Launch duy nhất.
* **Hiệu quả:** Tiết kiệm thời gian thao tác khi mượn xe 2 tiếng, loại bỏ hoàn toàn lỗi gõ nhầm lệnh trên Terminus khi thi đấu áp lực.

---

## 🚀 3. BẢNG CÔNG VIỆC CHI TIẾT (ACTIONABLE TODO-LIST)

### 🔴 Giai đoạn 1: Hợp nhất Codebase (Sprint 1 - Merge Code)
- [ ] **[Tùng & Lâm]** Đưa thuật toán xử lý ảnh Dual-Filter (HSV Red + White) và FSM né vật cản State-Aware từ `main_speed_track.py` của Tùng vào lớp `camera_processor.py` và `fsm_manager.py` của Lâm.
- [ ] **[Nhất & Tùng]** Thay thế P-Controller bằng `LQRController` trong bài Speed Track để kiểm thử độ mượt bẻ lái.
- [ ] **[Quyền & Tín]** Nhập gói `jetracer_smartcity` của Tín vào bộ khung `main_smart_city.py` của Quyền để chạy thử nhận diện biển báo qua TensorRT.

### 🟡 Giai đoạn 2: Tinh chỉnh Sa bàn & Cân chỉnh Parameter (Sprint 2 - Calibration)
- [ ] **[Tùng]** Kiểm thử thực tế bài Speed Track với sa bàn mới (Vạch biên đỏ + Lòng đường đen + Nền trắng), chốt giá trị `DODGE_OFFSET_PX` và tốc độ `BASE_SPEED`.
- [ ] **[Quyền & Lâm]** Kiểm thử luồng chạy Smart City qua các nút giao ngã tư, kiểm tra tính năng Re-planning khi cấm đường (`banned_edges`).
- [ ] **[Tín]** Kiểm tra độ chính xác nhận diện của `best.onnx` / TensorRT engine dưới điều kiện ánh sáng thực tế của phòng lab.
- [ ] **[Nhất]** Chạy bộ script test độc lập (`test_sensors.py`, `test_only_motors.py`) trên từng chiếc xe mới mượn để xác nhận phần cứng an toàn trước lượt chạy.

### 🟢 Giai đoạn 3: Chuẩn bị Thi đấu & Tự động hóa (Sprint 3 - Deployment)
- [ ] **[Cả đội]** Đóng gói file launch tổng hợp `production.launch`.
- [ ] **[Cả đội]** Viết script tự động dọn dẹp bộ nhớ/history ở cuối ca mượn xe (`clean_up.sh`) để bảo mật mã nguồn đội bóng.
- [ ] **[Cả đội]** Diễn tập chạy thử 2 lượt thi chính thức dưới áp lực thời gian (15 phút/lượt).

---

## 📋 4. KỊCH BẢN PHÂN CÔNG HỌP NHÓM (MEETING SYNC)

Khi họp nhóm hoặc nhắn tin cập nhật với 4 đồng đội, bạn có thể gửi đoạn tóm tắt sau:

> *"Chào mọi người, tui vừa checkout và review toàn bộ code của 4 nhánh (`lam`, `quyen`, `hnyat`, `tin`). Đội mình đang làm cực kỳ bài bản và mỗi người đang gánh 1 mảng rất mạnh! Tui đề xuất lộ trình hợp nhất code như sau:*
> 1. **Bài 1 (Speed Track):** Giữ core bám làn màu kép & FSM né vật cản của Tùng, ghép với bộ điều khiển **LQR của Nhất** để chạy tốc độ cao mượt nhất.
> 2. **Bài 2 (Smart City):** Dùng bộ khung FSM 8 bước & `MapNavigator` của **Quyền**, kết hợp với mô hình **YOLOv5n TensorRT (30 FPS) của Tín** và bộ nhận diện biển báo.
> 3. **Cấu trúc chung:** Đưa toàn bộ tham số cân chỉnh về file `settings.py` theo kiến trúc Phân lớp của **Lâm** để cả nhóm dễ tinh chỉnh khi ra sa bàn.
> 
> *Mọi người xem qua file `docs/TEAM_PROGRESS_TODO.md` tui vừa cập nhật để nhận task Giai đoạn 1 nhé!"*
