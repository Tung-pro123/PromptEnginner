# Kiến Trúc Các Tasks Điều Khiển Robot

Tài liệu này mô tả luồng dữ liệu (Data Pipeline) và vòng đời thực thi của hai chương trình cốt lõi giúp điều khiển Robot Jetson Nano đua xe: `ros_speed_track` và `ros_ai_navigation`. Cả hai đều được thiết kế dựa trên mẫu kiến trúc **Blackboard Pattern** để trao đổi dữ liệu.

---

## 1. Task: Speed Track (`ros_speed_track.py`)

Đây là bộ điều khiển cơ bản, phản xạ nhanh và ổn định nhất. Phù hợp cho mục tiêu đua xe tốc độ cao trên đường đua có vạch kẻ sẵn.

### Sơ đồ Luồng Dữ Liệu (Pipeline)
```mermaid
flowchart LR
    %% Style (Màu sắc và viền)
    classDef sensor fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef perceive fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef control fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef bb fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef debug fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef hw fill:#eceff1,stroke:#455a64,stroke-width:2px,color:#000

    %% Nodes
    Lidar["📡 Lidar Data"]:::sensor
    Cam["📷 Camera Data"]:::sensor
    BB[("Blackboard<br>(Shared Data)")]:::bb

    subgraph Perception["Perception Layer"]
        direction TB
        LP["LidarProcessor<br>- Tìm vật cản<br>- Đo khoảng cách"]:::perceive
        CP["CameraProcessor<br>- Tiền xử lý ảnh<br>- Phân đoạn làn"]:::perceive
    end

    subgraph StateControl["State & Control Layer"]
        direction TB
        FSM["⚙️ FSM Manager<br>- SAFE / DODGE"]:::control
        CTRL["🎛️ Controller<br>- Tính Steering<br>- Tính Throttle"]:::control
    end

    Motor["🚙 NvidiaRacecar<br>(Hardware)"]:::hw
    Debugger["🎥 Debugger<br>(Video & CSV)"]:::debug

    %% Connections
    Lidar --> LP
    Cam --> CP
    
    LP -->|"Dữ liệu vật cản"| BB
    CP -->|"Tọa độ waypoints"| BB
    
    BB -.->|"Đọc dữ liệu"| FSM
    FSM -.->|"Ghi DODGE/SAFE"| BB
    
    BB -.->|"Đọc State & Waypoints"| CTRL
    CTRL ===>|"Bơm thẳng lệnh"| Motor
    
    BB -.->|"Dữ liệu toàn cảnh"| Debugger
```

**Cách hoạt động (20Hz):**
1. Lidar và Camera đổ dữ liệu liên tục vào Blackboard.
2. FSM đọc cảm biến, quyết định xem xe đang ở trạng thái SAFE (an toàn) hay DODGE (phải né vật cản).
3. Controller đọc trạng thái FSM và Waypoints từ Camera, tính toán góc Steering phù hợp và truyền thẳng xuống động cơ.

---

## 2. Task: AI Navigation (`ros_ai_navigation.py`)

Đây là hệ thống có "Bộ Não" AI cấp cao. Ngoài việc phản xạ chạy theo vạch, nó có khả năng đưa ra các quyết định phức tạp như: rẽ ở ngã tư theo chỉ dẫn của biển báo, dừng lại khi gặp đèn đỏ, lùi xe khi bị kẹt, hoặc dừng khẩn cấp.

### Sơ đồ Luồng Dữ Liệu (Pipeline)
```mermaid
flowchart LR
    %% Style (Màu sắc và viền)
    classDef sensor fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef core fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef bb fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef ai fill:#fce4ec,stroke:#c2185b,stroke-width:3px,color:#000
    classDef hw fill:#eceff1,stroke:#455a64,stroke-width:2px,color:#000

    %% Nodes
    Lidar["📡 Lidar Data"]:::sensor
    Cam["📷 Camera Data"]:::sensor
    BB[("Blackboard<br>(Shared Data)")]:::bb

    subgraph Core["Core Processing System"]
        direction TB
        Perception["👁️ Perception<br>(Camera & Lidar)"]:::core
        Traffic["🚦 Traffic Detector<br>(Nhận diện Biển báo & Đèn)"]:::core
        FSM["⚙️ FSM Manager"]:::core
        CTRL["🎛️ Controller<br>(PID/Predictive)"]:::core
    end

    subgraph Brain["High-Level AI Brain"]
        AI{"🧠 AIDecisionEngine<br>Tuân thủ đèn giao thông<br>Rẽ theo biển báo<br>Xử lý kẹt xe"}:::ai
    end

    EXEC["⚡ _execute_command"]:::core
    Motor["🚙 NvidiaRacecar<br>(Hardware)"]:::hw

    %% Connections
    Lidar --> Perception
    Cam --> Perception
    Perception -->|"Cập nhật State"| BB
    
    Cam --> Traffic
    Traffic -->|"Đèn đỏ/xanh, Biển báo"| BB
    
    BB -.->|"Đọc dữ liệu"| FSM
    FSM -.->|"Ghi DODGE/SAFE"| BB
    
    BB -.->|"Đọc waypoints"| CTRL
    CTRL -.->|"Đề xuất Góc lái"| BB
    
    BB ===>|"Dữ liệu toàn cảnh"| AI
    AI ===>|"OVERRIDE (Rẽ, Dừng đèn đỏ, Lùi)"| BB
    
    BB -->|"Lấy Lệnh Cuối Cùng"| EXEC
    EXEC -->|"Điều khiển động cơ"| Motor
```

**Cách hoạt động (20Hz):**
1. Hệ thống Perception và FSM và Controller chạy tương tự như `Speed Track`.
2. Bổ sung thêm khối **Traffic Detector** phân tích ảnh Camera độc lập, tìm kiếm Đèn Giao Thông và Biển Báo rồi ghi kết quả vào Blackboard.
3. Controller chỉ đóng vai trò "Đề xuất góc lái". Lệnh này KHÔNG truyền trực tiếp xuống phần cứng.
4. Khối **`AIDecisionEngine`** sẽ đánh giá toàn bộ bức tranh theo thứ tự ưu tiên (Priority Chain):
   - **Mức 1 (Khẩn cấp):** Dừng ngay lập tức nếu vật cản quá gần.
   - **Mức 2 (Đèn giao thông):** Nếu nhận thấy `ĐÈN ĐỎ`, ép dừng xe (`WAIT_RED_LIGHT`).
   - **Mức 3 (Vật cản):** Kẹt cứng thì đứng chờ, đợi lâu quá thì lùi xe. Né tránh nếu có chỗ trống.
   - **Mức 4 (Ngã tư & Biển báo):** Nếu thấy ngã tư, AI sẽ kiểm tra xem trước đó có nhận diện được biển báo chỉ dẫn không. Nếu có biển báo Rẽ Trái/Phải/Thẳng, AI sẽ ép xe đi theo hướng đó.
   - **Mức 5 (Mặc định):** Cho phép lệnh của Controller đi qua để bám làn bình thường (`FOLLOW_LANE`).
5. Hàm `_execute_command` nhận lệnh cuối cùng từ AI và truyền xuống motor.
