# 📊 Phân Tích Proposal & Feedback Giám Khảo
## Team PromptEngineer — Jetson AI Racer Challenge 2026
> *Cập nhật: 27/06/2026*

---

## 1. Tổng Quan Điểm Số: 81.5/100 ✅ ĐẠT

| Hạng mục | Điểm tối đa | Điểm GK | Đánh giá |
|---|---|---|---|
| A. Tóm tắt & Tên đề tài | 10 | 8.5 | 🟢 Tốt |
| B. Bài toán, động lực, mục tiêu | 15 | 12.5 | 🟢 Tốt |
| C. Cơ sở lý thuyết & điểm mới | 10 | 9.0 | 🟢 Rất tốt |
| **D. Kiến trúc hệ thống** | **30** | **24.0** | **🟡 Cần cải thiện** |
| **E. Triển khai & quản lý rủi ro** | **15** | **11.0** | **🔴 Yếu nhất** |
| F. Kết quả dự kiến & đánh giá | 10 | 8.0 | 🟡 Cần bổ sung |
| G. Hình thức & tài liệu tham khảo | 10 | 8.5 | 🟢 Tốt |

> **Phần D (Kiến trúc) và E (Triển khai) bị trừ nhiều nhất** — đây là 2 phần cần tập trung cải thiện khi vào vòng chung kết.

### Điểm mạnh (Giám khảo khen):
- Related work, identified gap và đóng góp kỹ thuật trình bày tốt hơn nhiều proposal thông thường.
- Decision/planning cho obstacle avoidance và Smart City có FSM, công thức và baseline đánh giá rõ.
- Kế hoạch đánh giá có metric, baseline, telemetry/log và phân tích PID.
- Bài có cấu trúc học thuật, citation và lập luận kỹ thuật tương đối thuyết phục.

### Điểm cần cải thiện (Giám khảo chê):
- Phụ thuộc LiDAR và cơ chế in-place pivot/opposite wheel-pair actuation → cần xác minh với JetRacer
- Thiếu sơ đồ architecture tổng thể (chỉ có sơ đồ FSM cho Smart City)
- KPI cho CTE/ESR chưa có ngưỡng pass/fail
- Timeline thiếu phân công và buffer dự phòng
- **Cần chuyển các lệnh pivot/evasion sang mô hình lái thật nếu xe dùng Ackermann/servo steering**

---

## 2. ⚠️ VẤN ĐỀ QUAN TRỌNG NHẤT: JetBot vs JetRacer

Giám khảo nhấn mạnh: *"Cần xác minh tính hợp lệ của LiDAR và cơ chế pivot theo phần cứng JetRacer"*

### So sánh JetBot vs JetRacer

```
┌─────────────────────────────────────────────────────────────────────┐
│                         JETBOT                                      │
│  ┌──────┐                                          ┌──────┐        │
│  │ Bánh │  ←── Motor trái                          │ Bánh │        │
│  │ TRÁI │      (quay độc lập)                      │ PHẢI │        │
│  └──────┘                                          └──────┘        │
│           2 bánh quay ĐỘC LẬP → Quay tại chỗ ĐƯỢC                 │
│           Giống xe tăng / robot hút bụi                             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        JETRACER                                     │
│  ┌──────┐        SERVO (lái)         ┌──────┐                      │
│  │ Bánh │  ←── ↕ Góc lái             │ Bánh │                      │
│  │ TRƯỚC│      (servo motor)         │ TRƯỚC│                      │
│  └──────┘                            └──────┘                      │
│                                                                     │
│  ┌──────┐        MOTOR (ga)          ┌──────┐                      │
│  │ Bánh │  ←── Tốc độ                │ Bánh │                      │
│  │  SAU │      (throttle motor)      │  SAU │                      │
│  └──────┘                            └──────┘                      │
│           Lái như XE HƠI → KHÔNG quay tại chỗ được                  │
│           Giống xe RC / xe ô tô thật                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Bảng so sánh chi tiết

| Đặc điểm | JetBot (Differential Drive) | JetRacer (Ackermann Steering) |
|---|---|---|
| **Cơ cấu lái** | 2 motor độc lập, 2 bánh | 1 servo lái (trước) + 1 motor ga (sau) |
| **Quay tại chỗ** | ✅ CÓ THỂ (quay 2 bánh ngược chiều) | ❌ KHÔNG THỂ (phải tiến/lùi để rẽ) |
| **Bán kính quay nhỏ nhất** | 0 (quay tại chỗ) | ~20-30cm (phụ thuộc góc lái servo) |
| **Điều khiển code** | `robot.set_motors(left, right)` | `car.steering = góc` + `car.throttle = tốc_độ` |
| **Rẽ 90°** | Đứng yên, quay 2 motor ngược chiều | Phải vừa đi vừa đánh lái (giống lái xe hơi) |
| **API Python** | `from jetbot import Robot` | `from jetracer.nvidia_racecar import NvidiaRacecar` |
| **Tốc độ** | Chậm (~0.2-0.3 m/s) | Nhanh hơn (~0.5-1.0 m/s) |
| **Phù hợp với** | Sa bàn nhỏ, Smart City (rẽ giao lộ) | Đường đua tốc độ (Speed Track) |

### NẾU LÀ JETRACER, cần thay đổi gì trong code?

#### A. Thay đổi phần Hardware Init

```python
# ❌ CODE CŨ (JetBot)
from jetbot import Robot
robot = Robot()
robot.set_motors(0.3, 0.3)       # 2 bánh cùng tốc độ = đi thẳng
robot.set_motors(0.2, -0.2)      # 2 bánh ngược chiều = quay tại chỗ

# ✅ CODE MỚI (JetRacer)
from jetracer.nvidia_racecar import NvidiaRacecar
car = NvidiaRacecar()
car.steering = 0.0               # Thẳng (range: -1.0 trái ↔ +1.0 phải)
car.throttle = 0.3               # Ga (range: -1.0 lùi ↔ +1.0 tiến)
```

#### B. Thay đổi logic rẽ

```python
# ❌ CODE CŨ (JetBot) - Quay tại chỗ
def turn_robot(self, degrees):
    if degrees > 0:
        self.robot.set_motors(0.2, -0.2)   # Quay phải tại chỗ
    else:
        self.robot.set_motors(-0.2, 0.2)   # Quay trái tại chỗ
    time.sleep(abs(degrees) / 90 * 0.8)
    self.robot.stop()

# ✅ CODE MỚI (JetRacer) - Phải vừa đi vừa rẽ
def turn_robot(self, degrees):
    if degrees > 0:
        self.car.steering = 0.8            # Đánh lái phải
    else:
        self.car.steering = -0.8           # Đánh lái trái
    self.car.throttle = 0.15               # Đi chậm trong khi rẽ
    time.sleep(abs(degrees) / 90 * 1.2)    # Thời gian lâu hơn vì phải đi vòng
    self.car.steering = 0.0                # Trả lái thẳng
    self.car.throttle = 0.0                # Dừng
```

#### C. Thay đổi logic bám line

```python
# ❌ CODE CŨ (JetBot) - Điều chỉnh bằng chênh lệch tốc độ 2 bánh
def correct_course(self, error):
    adj = error * CORRECTION_GAIN
    self.robot.set_motors(BASE_SPEED + adj, BASE_SPEED - adj)

# ✅ CODE MỚI (JetRacer) - Điều chỉnh bằng góc lái servo
def correct_course(self, error):
    steering_adj = error * STEERING_GAIN   # Tính góc lái
    steering_adj = max(-1.0, min(1.0, steering_adj))  # Giới hạn -1 → +1
    self.car.steering = steering_adj
    self.car.throttle = BASE_THROTTLE
```

#### D. Approach 1 (Né vật cản) phải thay đổi hoàn toàn

```
JetBot (quay tại chỗ):        JetRacer (phải đi vòng):
                                
  ║ ║  ──→ ┌─┐                  ║ ║  ──→  ╭───╮
  ║ ║      │█│   Dừng            ║ ║       │ █ │  Chạy chậm
  ║ ║  quay│█│   Quay 90°        ║ ║       │ █ │  + đánh lái
  ║ ║  90° │█│   Đi ngang        ║ ║     ╭─╯ █ │  đi vòng qua
  ║ ║  ──→ │█│   Quay -90°       ║ ║  ──→╰────╯  vật cản
  ║ ║      └─┘   Đi thẳng       ║ ║              rồi trả lái
```

**Kết luận:** Nếu là JetRacer, không thể dùng chiến thuật "dừng → quay 90° → đi ngang" được. Phải dùng chiến thuật **đi vòng cung** (arc) quanh vật cản, tương tự Approach 2 (Trigonometric) trong Proposal.

---

## 3. GAP: Proposal vs. Code Thực Tế

| # | Proposal đề xuất | Code hiện tại | Trạng thái |
|---|---|---|---|
| 1 | **Perspective Transform (Bird's Eye View)** | ❌ Không có, chỉ dùng HSV filter | 🔴 THIẾU |
| 2 | **PID Controller** (Kp, Ki, Kd) | ❌ Chỉ có P-controller (`CORRECTION_GAIN`) | 🟡 SƠ KHAI |
| 3 | **Approach 1: Orthogonal Evasion** | ❌ Không có code né vật cản | 🔴 THIẾU |
| 4 | **Approach 2: Trigonometric Evasion** | ❌ Không có | 🔴 THIẾU |
| 5 | **Smart City FSM 4 trạng thái** (S1-S4) | ❌ FSM hiện tại khác hoàn toàn (đơn giản hơn) | 🟡 KHÁC |
| 6 | **Template Matching** biển báo | ❌ Không có | 🔴 THIẾU |
| 7 | **YOLOv8-nano + TensorRT** | ❌ Dùng Roboflow API (HTTP, cần internet) | 🟡 THAY THẾ |
| 8 | **Traffic Light Handling** | ❌ Không có code xử lý đèn giao thông | 🔴 THIẾU |
| 9 | **Kalman Filter** cho LiDAR | ❌ Không có | 🔴 THIẾU |
| 10 | **Telemetry Dashboard** | 🟡 Có MQTT publish nhưng chưa có dashboard | 🟡 MỘT PHẦN |
| 11 | **Canny Edge Detection** | ❌ Chỉ dùng HSV color masking | 🟡 KHÁC |
| 12 | **Dual-ROI** | ✅ Có Proximal + Distal ROI | 🟢 ĐÃ CÓ |
| 13 | **Map Navigation** (A*/Dijkstra) | ✅ Có map_navigator.py | 🟢 ĐÃ CÓ |
| 14 | **LiDAR Intersection Detection** | ✅ Có opposite_detector.py | 🟢 ĐÃ CÓ |

> **Tỷ lệ hoàn thành so với Proposal: ~30-40%.** Nhiều feature cốt lõi chưa implement.

---

## 4. 🗺️ Lộ Trình Hành Động

### Giai đoạn 0: XÁC MINH (Tối nay khi mượn xe) ⚡ CRITICAL

- [ ] **Xe là JetBot hay JetRacer?** → Quyết định toàn bộ logic control
- [ ] **Có LiDAR trên xe không?** → Quyết định Approach 1 & 2 có khả thi không
- [ ] **Có internet khi thi không?** → Quyết định Roboflow API hay local YOLO
- [ ] **SSH/VNC vào Jetson được không?** IP bao nhiêu?
- [ ] **Sa bàn trông như nào?** Màu vạch, kích thước, biển báo, đèn giao thông?

---

### Giai đoạn 1: SỬA LỖI NGHIÊM TRỌNG (Ngày 1-2) 🔴

| Việc | Chi tiết |
|---|---|
| Viết lại `turn_robot()` | Phù hợp cơ cấu lái thực tế (JetBot HOẶC JetRacer) |
| Viết lại `correct_course()` | Nâng cấp từ P-controller → PID đầy đủ |
| Viết lại Hardware Init | `from jetbot import Robot` → `from jetracer...` (nếu JetRacer) |
| Vẽ sơ đồ kiến trúc tổng thể | Sensor → Perception → Decision → Control → Actuator |
| Định nghĩa KPI cụ thể | CTE < X pixels, ESR > Y%, FPS ≥ 20 |

---

### Giai đoạn 2: SPEED TRACK - 30% điểm (Ngày 2-4) 🏁

| Việc | Chi tiết |
|---|---|
| Thêm Perspective Transform | Bird's Eye View cho lane detection chính xác hơn |
| Implement né vật cản | Approach 1 (nếu JetBot) hoặc Arc-based (nếu JetRacer) |
| Thêm Kalman filter LiDAR | Lọc nhiễu sensor |
| Calibrate tốc độ & góc trên xe thật | BASE_SPEED, TURN_DURATION, STEERING_GAIN |

---

### Giai đoạn 3: SMART CITY - 40% điểm (Ngày 3-5) 🏙️

| Việc | Chi tiết |
|---|---|
| Implement FSM mới (S1→S4) | Đúng theo Proposal đã nộp |
| Thêm Traffic Light Detection | Nhận diện đèn đỏ/xanh bằng HSV hoặc YOLO |
| Quyết định perception engine | Roboflow API (online) hay YOLOv8-nano (local)? |
| Thêm Template Matching fallback | Backup khi YOLO/Roboflow fail |
| Cập nhật map.json | Theo sa bàn chính thức 2026 |

---

### Giai đoạn 4: TÍCH HỢP & TEST (Ngày 5-7) 🔧

| Việc | Chi tiết |
|---|---|
| Tích hợp cả 2 bài | Speed Track + Smart City chạy trên 1 xe |
| Stress test 10+ lần | Chạy liên tục, ghi log, phân tích video debug |
| PID tuning | Dựa trên log thực tế |
| Dashboard (nếu kịp) | MQTT → Web dashboard giám sát real-time |

---

## 5. Phân Công Đề Xuất (5 thành viên)

| Thành viên | Vai trò | Nhiệm vụ chính |
|---|---|---|
| **Lê Thanh Tùng** | Leader / Integration | Tích hợp, test trên xe, sơ đồ kiến trúc, quản lý git |
| **Lô Thái Quyên** | Perception | Camera processing, Bird's Eye View, nhận diện biển báo/đèn |
| **Huỳnh Nhật** | Planning | FSM mới (S1-S4), map navigation, Smart City logic |
| **Nguyễn Trung Tín** | Control | PID controller, `turn_robot()` mới, calibrate trên xe |
| **Nguyễn Đức Bảo Lâm** | Speed Track | Obstacle evasion (Approach 1 & 2), Kalman filter LiDAR |

---

## 6. Checklist Tối Nay (Khi Mượn Xe)

```
1️⃣  Xác minh: JetBot hay JetRacer? → ẢNH HƯỞNG TOÀN BỘ CODE
2️⃣  Xác minh: Có internet khi thi? → Roboflow API hay local YOLO?  
3️⃣  SSH vào Jetson → pull code → chạy thử cơ bản
4️⃣  Ghi lại: IP Jetson, password, phiên bản Python/ROS/JetPack
5️⃣  Chụp ảnh sa bàn nếu có → để cập nhật map.json
```

---

*File này được tạo tự động. Đọc kỹ trước khi bắt đầu code!*
