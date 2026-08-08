# 📘 Báo Cáo Review Mã Nguồn - Nhánh `hnyat` (Nhất)

Tài liệu này tổng hợp phân tích chi tiết mã nguồn, thuật toán điều khiển tối ưu LQR và giải pháp kỹ thuật trên nhánh **`hnyat`** (Nhất), đồng thời so sánh với phương án của nhánh **`tung`**.

---

## 1. TỔNG QUAN GIẢI PHÁP KỸ THUẬT
Nhánh `hnyat` tập trung xây dựng **Core Bộ Điều Khiển Tối Ưu LQR (Linear Quadratic Regulator)** dựa trên mô hình động học xe đạp và thiết lập **Bộ kịch bản kiểm thử toàn diện (Test Suite)**.

```
       [ Input: C_near, C_far from Camera ]
                        │
                        ▼
    [ Kinematic Bicycle Model (State Matrix A, B) ]
                        │
                        ▼
    [ Discrete Algebraic Riccati Equation (DARE) ]
                        │
                        ▼
     [ Feedback Gain K = [K1, K2, K3, K4] ] ──> [ Optimal Steering Command ]
```

---

## 2. CHI TIẾT THUẬT TOÁN & MÔ-ĐUN NỔI BẬT

### 🏎️ A. Bộ điều khiển Tối ưu LQR (`src/core/control/lqr_controller.py`)
1. **Mô hình Xe đạp Động học (Kinematic Bicycle Model):**
   * Chiều dài cơ sở xe (Wheelbase): $L = 0.18$m.
   * Hệ số quy đổi pixel $\rightarrow$ mét thực tế: $\text{scale\_factor} = 0.0015$ ($1\text{px} \approx 1.5\text{mm}$).
   * Vectơ 4 trạng thái phạt: $X = [e, \dot{e}, e_\theta, \dot{e}_\theta]^T$
     * $e$: Sai số lệch tâm đường (Lateral error).
     * $\dot{e}$: Tốc độ thay đổi sai số lệch tâm.
     * $e_\theta$: Sai số góc hướng xe so với độ cong đường (Heading error).
     * $\dot{e}_\theta$: Tốc độ thay đổi góc hướng (Yaw rate).
2. **Giải phương trình Riccati Đại số Rời rạc (DARE Solver):**
   * Hàm `solve_DARE(A, B, Q, R)` tự động lặp Riccati để tính ma trận Gain hồi tiếp $K = [K_1, K_2, K_3, K_4]$ tối ưu theo thời gian thực.
   * **Ma trận phạt trạng thái $Q$:** $\text{diag}([15.0, 1.0, 8.0, 0.5])$ (Phạt nặng sai số khoảng cách và sai số góc).
   * **Ma trận phạt góc lái $R$:** $[[1.2]]$ (Hạn chế tối đa việc đánh lái gấp lật bánh).
3. **Cơ chế Dịch vạch Tiếp tuyến Mượt mà (S-Curve Ramping):**
   * Phương thức `update_offset(dt)` tự động dịch vạch ảo mượt mà với vận tốc tiếp tuyến $0.35$ m/s, giúp xe chuyển làn S-Curve tự nhiên.

### 🧪 B. Bộ Kịch bản Kiểm thử Chuyên sâu (`tests/`)
Nhất đã xây dựng tập hợp các script test từng thành phần cảm biến độc lập trước khi ráp lên xe thật:
* `test_only_camera.py`: Test riêng luồng ảnh camera và hiển thị các đường ROI quét vạch.
* `test_only_lidar.py`: Test riêng cảm biến LiDAR và vẽ bản đồ radar.
* `test_only_motors.py`: Test đáp ứng của servo góc lái và động cơ ga.
* `test_obstacle_avoidance.py`: Mô phỏng và kiểm thử khả năng lách né vật cản động bằng LiDAR.
* `test_speed_track_concept.py`: Thử nghiệm toàn bộ thuật toán Speed Track trên môi trường giả lập.

---

## 3. SO SÁNH GIỮA NHÁNH `hnyat` VÀ NHÁNH `tung` (BẠN)

| Tiêu chí | Nhánh của **TÙNG** (`tung`) | Nhánh của **NHẤT** (`hnyat`) |
| :--- | :--- | :--- |
| **Trọng tâm** | **Speed Track Thực tế:** Xử lý ảnh màu kép (HSV Red + White), phân loại biên State-Aware và dứt điểm lượt chạy trên sa bàn. | **Lý thuyết Điều khiển & Test Suite:** Xây dựng toán học LQR, giải phương trình Riccati và viết bộ script test đơn vị (Unit tests). |
| **Bộ điều khiển Lái** | **P-Controller + Safety Override:** Góc bẻ lái tỷ lệ $P = 0.007$, khóa kịch lái tối thiểu $0.28$ khi né và trả làn 2 giai đoạn (Open-loop $0.50$ trong 1.2s). | **Optimal LQR Controller:** Tự động giải DARE tính Gain $K$ phạt sai số khoảng cách $e$ và sai số góc $e_\theta$ theo mô hình xe đạp động học. |
| **Hệ thống Test** | Test trực tiếp trên xe thực tế và ghi video/CSV debug (`speed_track_run.avi`). | Có bộ script test mô phỏng riêng biệt từng cảm biến (`test_only_camera`, `test_only_lidar`, `test_only_motors`). |

---

## 4. ĐÁNH GIÁ CHUNG & ĐỀ XUẤT TÍCH HỢP

* **Ưu điểm lớn nhất của Nhất:** Thuật toán **LQR được xây dựng cực kỳ chuẩn xác về mặt toán học và lý thuyết điều khiển**, giúp xe vào cua và né vật cản theo đường cong mượt mà hơn nhiều so với bộ điều khiển P thông thường. Đồng thời bộ script `tests/` rất hữu ích để kiểm tra phần cứng.
* **Đề xuất tích hợp:** 
  * Bạn có thể giữ nguyên luồng FSM và xử lý ảnh Dual-Filter của bạn (`tung`), sau đó **thay bộ điều khiển P-Controller hiện tại bằng lớp `LQRController` của Nhất** (`from src.core.control.lqr_controller import LQRController`). Sự kết hợp này sẽ giúp xe vừa không bao giờ đâm biên, vừa lượn cua êm ái như xe tự hành cao cấp!
