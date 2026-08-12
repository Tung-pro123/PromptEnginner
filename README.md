# Jetson AI Racer Challenge 2026 🏎️🤖
## Team: PromptEngineer

> **Cuộc thi:** Jetson AI Racer Challenge 2026 - FPT Education  
> **Nền tảng:** NVIDIA Jetson Nano + JetRacer/JetBot  
> **Ngôn ngữ:** Python 3 + ROS (Robot Operating System)

---

## 📁 Cấu Trúc Thư Mục

```
Jetson/
├── docs/                          # 📄 Tài liệu cuộc thi
│   ├── Thể lệ.docx.pdf           #   Thể lệ & luật thi chính thức
│   ├── Đề bài chi tiết.docx.pdf   #   Đề bài chi tiết (Speed Track + Smart City)
│   ├── PromptEngineer_...pdf       #   Proposal của đội mình đã nộp
│   └── 03 - PromptEngineer.pdf    #   Feedback từ giám khảo về proposal
│
├── src/                           # 🧑‍💻 SOURCE CODE CHÍNH (code ở đây!)
│   ├── core/                      #   🔧 Module dùng chung cho cả 2 bài thi
│   │   ├── perception/            #     👁️ Xử lý Camera & LiDAR (Dual-ROI, HSV filter)
│   │   │                          #     → TODO: Tách logic xử lý ảnh từ main vào đây
│   │   ├── control/               #     🎮 Điều khiển động cơ (PID, bám line, né vật cản)
│   │   │                          #     → TODO: Tách logic correct_course(), turn_robot() vào đây
│   │   ├── planning/              #     🗺️ Điều hướng & tìm đường
│   │   │   ├── map_navigator.py   #       Thuật toán A* / Dijkstra tìm đường ngắn nhất trên đồ thị
│   │   │   └── callmap.py         #       Gọi API server để lấy bản đồ mới nhất (map.json)
│   │   └── utils/                 #     🛠️ Tiện ích & cấu hình
│   │       ├── map.json           #       File bản đồ sa bàn (đồ thị các node + cạnh)
│   │       └── opposite_detector.py #     Phát hiện giao lộ bằng LiDAR (đối xứng 2 bên)
│   │
│   ├── speed_track/               #   🏁 BÀI THI 1: SPEED TRACK (30% điểm)
│   │   └── main_speed_track.py    #     File chạy chính - bám lane, né vật cản, checkpoint
│   │                              #     → Dựa trên problem_a cũ
│   │
│   └── smart_city/                #   🏙️ BÀI THI 2: SMART CITY (40% điểm)
│       └── main_smart_city.py     #     File chạy chính - FSM, biển báo, QR, điều hướng
│                                  #     → Dựa trên problem_b cũ (có Roboflow API)
│
├── archive/                       # 📦 Code cũ từ Hackathon (backup, KHÔNG SỬA Ở ĐÂY)
│   ├── problem_a/                 #   Bài A cũ (line following + map navigation)
│   ├── problem_b/                 #   Bài B cũ (biển báo + Roboflow + submit API)
│   └── problem_c/                 #   Bài C cũ (tương tự A)
│
├── restructure.py                 # 🔨 Script tự động tái cấu trúc (đã chạy xong, giữ lại để tham khảo)
├── .gitignore                     # Git ignore
└── README.md                      # 📖 File này
```

---

## 🏁 2 Bài Thi Chính

### Bài 1: Speed Track (30% điểm tổng)
**File chính:** `src/speed_track/main_speed_track.py`

| Yêu cầu | Mô tả |
|---|---|
| Bám lane | Robot đi theo vạch kẻ đường bằng camera (HSV filter + Dual ROI) |
| Né vật cản | Phát hiện và tránh vật cản trên đường đua |
| Checkpoint | Đi qua đúng các checkpoint trên sa bàn |
| Tốc độ | Hoàn thành đường đua nhanh nhất có thể |

### Bài 2: Smart City (40% điểm tổng)
**File chính:** `src/smart_city/main_smart_city.py`

| Yêu cầu | Mô tả |
|---|---|
| Điều hướng | Đi từ Start → End theo đường ngắn nhất trên sa bàn đồ thị |
| Nhận diện biển báo | Dùng Roboflow API nhận diện biển chỉ dẫn (N/E/S/W) và biển cấm (NN/NE/NS/NW) |
| Xử lý giao lộ | FSM (Finite State Machine) quản lý trạng thái tại mỗi giao lộ |
| QR Code / Toán | Đọc QR code hoặc giải toán tại các node đặc biệt |
| Submit kết quả | Gửi kết quả nhận diện lên server qua API |

---

## 🚀 Hướng Dẫn Chạy Nhanh

### Chuẩn bị trên Jetson
```bash
# 1. Clone repo
git clone <repo_url>
cd Jetson

# 2. Bật LiDAR
roslaunch jetracer lidar.launch

# 3. Bật Camera (mở terminal mới)
roslaunch jetracer csi_camera.launch

# 4a. Chạy Speed Track (mở terminal mới)
cd src/speed_track
python3 main_speed_track.py

# 4b. HOẶC chạy Smart City
cd src/smart_city
python3 main_smart_city.py
```

### Cập nhật bản đồ mới từ server
```bash
cd src/core/planning
python3 callmap.py
```

---

## 🔧 Cấu Hình Quan Trọng

### Biến môi trường cần thiết (cho Smart City)
```bash
export ROBOFLOW_API_KEY="<api_key_của_đội>"    # API key Roboflow để nhận diện biển báo
export TEAM_NAME="PromptEngineer"               # Tên đội
export SUBMIT_URL="<url_submit_server>"         # URL server nộp kết quả
```

### Tham số có thể điều chỉnh (trong `setup_parameters()`)
| Tham số | Mô tả | Giá trị mặc định |
|---|---|---|
| `BASE_SPEED` | Tốc độ di chuyển cơ bản | Speed: 0.27 / Smart: 0.16 |
| `TURN_SPEED` | Tốc độ khi quay | 0.2 |
| `TURN_DURATION_90_DEG` | Thời gian quay 90° (giây) | 0.8 |
| `CORRECTION_GAIN` | Hệ số hiệu chỉnh bám line | 0.5 |
| `LINE_COLOR_LOWER/UPPER` | Ngưỡng HSV phát hiện vạch kẻ | [0,0,0] / [180,255,75] |
| `YOLO_CONF_THRESHOLD` | Ngưỡng tin cậy nhận diện | 0.6 |

---

## 📋 TODO - Phân Công Công Việc

### 🔴 Ưu tiên cao (Làm trước khi test trên xe)
- [ ] Cập nhật `map.json` theo sa bàn chính thức cuộc thi 2026
- [ ] Điền `ROBOFLOW_API_KEY` và kiểm tra model nhận diện biển báo mới
- [ ] Cập nhật token trong `callmap.py` (token API server cuộc thi mới)
- [ ] Calibrate lại `TURN_DURATION_90_DEG` và `BASE_SPEED` trên xe thật

### 🟡 Ưu tiên trung bình (Tối ưu hóa)
- [ ] Tách logic xử lý ảnh (HSV, ROI) từ `main_*.py` ra `src/core/perception/`
- [ ] Tách logic điều khiển motor (`correct_course`, `turn_robot`) ra `src/core/control/`
- [ ] Thêm logic xử lý đèn giao thông (nếu có trong đề thi)
- [ ] Tối ưu PID controller cho bám line mượt hơn

### 🟢 Ưu tiên thấp (Nice to have)
- [ ] Thêm dashboard MQTT để giám sát robot real-time
- [ ] Cải thiện logic `stabilize_after_turn()` sau khi rẽ
- [ ] Thêm unit tests cho `map_navigator.py`

---

## 🧠 Kiến Trúc FSM (Finite State Machine)

```
WAITING_FOR_LINE → DRIVING_STRAIGHT → APPROACHING_INTERSECTION
                        ↑                       ↓
                  REACQUIRING_LINE ← LEAVING_INTERSECTION ← HANDLING_EVENT
                        ↓
                    DEAD_END / GOAL_REACHED
```

| Trạng thái | Mô tả |
|---|---|
| `WAITING_FOR_LINE` | Chờ camera nhìn thấy vạch kẻ đường |
| `DRIVING_STRAIGHT` | Đang bám line, kiểm tra LiDAR + ROI dự báo |
| `APPROACHING_INTERSECTION` | Tiến vào trung tâm giao lộ |
| `HANDLING_EVENT` | Dừng, nhận diện biển báo, quyết định hướng đi |
| `LEAVING_INTERSECTION` | Đi thẳng thoát khỏi giao lộ |
| `REACQUIRING_LINE` | Tìm lại vạch kẻ đường sau khi rẽ |
| `GOAL_REACHED` | Đã đến đích, dừng robot |
| `DEAD_END` | Lỗi hoặc không tìm được đường, dừng robot |

---

## ⚠️ Lưu Ý Quan Trọng

1. **KHÔNG sửa code trong `archive/`** - Đó là backup code cũ. Mọi thay đổi code trong `src/`.
2. **Luôn test trên xe thật** trước khi nộp - Các tham số tốc độ, góc quay rất phụ thuộc vào phần cứng.
3. **Kiểm tra pin** trước mỗi lần chạy - Pin yếu ảnh hưởng đến tốc độ motor và kết quả bám line.
4. **Ghi video debug** - Robot tự ghi video `jetbot_run.avi` mỗi lần chạy, dùng để phân tích lỗi.
5. **Map mới** - Nhớ chạy `callmap.py` để lấy map mới trước mỗi lượt thi (BTC có thể đổi map).

---

## 👥 Thành Viên Nhóm

| Vai trò | Nhiệm vụ chính |
|---|---|
| **Leader / Tích hợp** | Ghép các module, test trên xe, quản lý git |
| **Perception** | Camera processing, LiDAR, nhận diện biển báo (Roboflow) |
| **Planning / Control** | Thuật toán tìm đường, PID tuning, điều khiển motor |
| **Infrastructure** | MQTT dashboard, API submit, logging & debug |

---

*Last updated: 26/06/2026*
