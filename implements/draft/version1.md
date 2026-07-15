# Báo Cáo Tổng Kết Phiên Bản: Speed Track v1 (Blackboard Pattern)

Báo cáo này tóm tắt toàn bộ những thay đổi, cải tiến và tái cấu trúc hệ thống đã được thực hiện cho module điều khiển bám đường và né vật cản (`speed_track1`) của Robot Jetson.

## 1. Tái Cấu Trúc Kiến Trúc (Architecture Refactoring)

* **Tách biệt và chuẩn hoá tên gọi**: 
  - Đổi tên `racer_controller.py` thành `pid_controller.py` và đổi tên class thành `PIDController` để phản ánh đúng bản chất giải thuật điều khiển (Proportional-Integral-Derivative).
  - Loại bỏ các file trùng lặp, dư thừa trước đó.
* **Tổ chức lại thư mục**: 
  - Chuyển thư mục `tasks/` (chứa node khởi chạy ROS `ros_speed_track.py`) ra khỏi thư mục `src/` nhằm tuân thủ thiết kế chuẩn của một project Python/ROS lớn. 
  - Sửa lại toàn bộ các đường dẫn import (`sys.path`) để hệ thống chạy ổn định.
* **Chuyển đổi sang Blackboard Pattern**:
  - Tách rời hoàn toàn phần dữ liệu và phần xử lý. Khởi tạo `src/core/blackboard.py` để đóng vai trò làm không gian lưu trữ dữ liệu tập trung (Knowledge Base).
  - Biến đổi các module (`LidarProcessor`, `CameraProcessor`, `FSMManager`, `PIDController`, `Debugger`) thành các Knowledge Sources. Mỗi module chỉ việc nhận biến `blackboard` vào, đọc các tham số cần thiết, tính toán và ghi ngược lại kết quả (ví dụ: `steering`, `center_x`, `state_name`) vào blackboard.
  - Vòng lặp chính trong `tasks/ros_speed_track.py` nay cực kỳ tinh gọn, chỉ đóng vai trò gọi tuần tự hàm `.process(blackboard)` của từng module.

## 2. Xử Lý Sensor & ROS Subscribers

* Các hàm ROS callback (như xử lý bản tin `/scan` và `/csi_cam_0/image_raw`) được rút khỏi file main và đưa sâu vào bên trong chính các Processor (`lidar_processor.py`, `camera_processor.py`).
* Node chính giờ đây chỉ truyền thẳng tham chiếu của Processor vào Subscriber:
  ```python
  rospy.Subscriber('/scan', LaserScan, self.lidar.ros_callback)
  rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera.ros_callback)
  ```
* Hàm callback tự động cập nhật dữ liệu (`latest_image`, `latest_scan`) vào thẳng Blackboard theo thời gian thực dưới nền (background).

## 3. Hệ Thống Log & Debug (Session-based Logging)

* **Phân tách theo Phiên (Session)**: 
  - Hệ thống log nay tự động gom nhóm dữ liệu theo từng lần chạy vào thư mục `logs/session_X/` (với X tự động tăng).
  - Tự động sinh file `session_info.txt` ghi nhận timestamp khởi động và kết thúc của robot.
* **Ghi hình Video (VideoWriter)**:
  - Bổ sung khả năng record video ra định dạng MP4 để có thể xem lại sau phiên chạy.
  - `camera_log.mp4`: Ghi lại raw video (ảnh gốc) thu được từ camera trước của xe.
  - `lidar_log.mp4`: Tích hợp thuật toán tự động render (visualize) mảng dữ liệu quét Lidar thành dạng đồ hoạ không gian (gồm tâm xe, lưới toạ độ mét, và các chấm trắng mô phỏng vật cản xung quanh) lên khung hình đen.
* **Dữ liệu thô**: 
  - Vẫn duy trì xuất file `speed_track_debug.csv` chi tiết từng frame (State, Khoảng cách vật cản, Góc vật cản, Offset, Góc lái).

## 4. Kiểm Thử & Mô Phỏng (Mock Testing)

* Bổ sung kịch bản Unit Test toàn diện tại `tests/test_blackboard_flow.py`.
* Tạo một môi trường giả lập hoàn toàn bằng các ma trận mảng numpy (`np.zeros`) mô phỏng hình ảnh có vạch kẻ đường, cũng như tạo các mảng vector giả lập tín hiệu tia Lidar bị dội lại.
* **Test Case 1 (Normal Driving)**: Mô phỏng xe chạy trên đường thẳng bình thường, chứng minh hệ thống tính đúng tâm đường, đưa ra góc lái cân bằng (0.0).
* **Test Case 2 (Dodging)**: Đưa vật cản nhân tạo vào vùng quét Lidar bên trái. FSM tự động chuyển sang state DODGING và ra quyết định chuyển vạch ảo (offset) sang bên phải, bẻ lái thành công.
* Kết quả test 100% Pass, chứng minh kiến trúc Blackboard tích hợp hoạt động mượt mà và không còn phụ thuộc cứng vào phần cứng Jetson hay ROS master.
