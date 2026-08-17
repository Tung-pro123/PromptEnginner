# 🏎️ Autonomous Line Following & Obstacle Avoidance via Online Imitation Learning (DAgger)

Hệ thống điều khiển xe tự hành kết hợp **Bám làn đường (Line Following)** và **Né vật cản (Obstacle Avoidance)** sử dụng phương pháp **Học tương tác trực tuyến (Interactive / Online Imitation Learning - DAgger)**. 

Mô hình AI cho phép xe vừa tự điều khiển (Inference), vừa thu thập dữ liệu can thiệp sửa lỗi từ tay cầm Gamepad (`Joy`) và cập nhật trọng số trực tuyến (Online Model Updating) để tối ưu hóa quỹ đạo chuyển động.

---

## 📌 Tính năng nổi bật (Key Features)

- **Xử lý đa cảm biến (Multi-Sensor Fusion):** Kết hợp các đặc trưng bám line từ Camera và vector khoảng cách phân vùng từ LiDaR ($180^\circ$).
- **Học trực tuyến DAgger:** Cho phép con người dùng Joy can thiệp trực tiếp khi AI lái lệch/gặp nguy hiểm. Dữ liệu sửa lỗi được đẩy thẳng vào Replay Buffer để train ngay lập tức.
- **Kiến trúc Đa tiến trình (Multi-Threading / Multi-Processing):** Tách biệt vòng lặp điều khiển thời gian thực ($30\text{–}50\text{ Hz}$) và tiến trình huấn luyện ngầm (Background Training), tránh tình trạng giật lag phần cứng.
- **Chống quên kiến thức cũ (Experience Replay):** Trộn dữ liệu can thiệp mới với tập dữ liệu anchor cơ bản để tránh hiện tượng *Catastrophic Forgetting*.
- **Lớp an toàn chủ động (Hybrid Safety Layer):** Bộ lọc đệm can thiệp phanh/ngắt ga tức thì khi khoảng cách LiDaR chạm ngưỡng nguy hiểm ($d < d_{\text{critical}}$).

---

## 🏗️ Kiến trúc Hệ thống (System Architecture)

### 1. Vector Trạng thái Đầu vào (Input State Vector $S_t$)

Vector đầu vào gọn nhẹ giúp tối ưu hóa thời gian tính toán trên các thiết bị nhúng (Raspberry Pi / Jetson Nano):

$$S_t = \left[ e_y, \theta_e, \text{line\_visible}, d_1, d_2, \dots, d_N \right]$$

- **Vision Features:**
  - `e_y` ($e_y$): Độ lệch tâm so với line.
  - `heading_error` ($\theta_e$): Góc lệch hướng so với line.
  - `line_visible`: Cờ trạng thái nhận diện line ($1$: nhìn thấy, $0$: bị vật cản che khuất).
- **LiDAR Features ($d_1 \dots d_N$):** Khoảng cách ngắn nhất trong $N$ vùng quét phía trước ($180^\circ$), chuẩn hóa về $[0.0, 1.0]$.

### 2. Đầu ra Điều khiển (Action Targets)
- `steer`: Góc đánh lái trong khoảng $[-1.0, 1.0]$.
- `throttle`: Tay ga / tốc độ trong khoảng $[0.0, 1.0]$.

---

## 🔄 Luồng Điều khiển & Can thiệp (Control Flow)

```text
[Camera/LiDAR] ---> [Tính Vector S_t] 
                            |
                     (Kiểm tra Joy)
                            |
         +------------------+------------------+
         |                                     |
  (Không đụng Joy)                      (Đang nhích Joy)
         |                                     |
  [AI Control Mode]                   [Human Intervention]
  - Xe chạy theo Model                - Nhượng quyền lái cho Joy
  - A_t = Model_AI(S_t)               - A_t = A_Joy
                                      - LƯU: Push (S_t, A_Joy) 
                                        vào Replay Buffer
                                               |
                                               v
                                      [Background Trainer]
                                      - Cập nhật trọng số AI