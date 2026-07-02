#!/usr/bin/env python3
"""
Jetson AI Racer Challenge 2026 - Speed Track (Bài 1)
Hệ thống điều khiển hoàn toàn tự động không sử dụng bản đồ (Mapless Autonomous).
- Sử dụng thuật toán bám tâm đường bằng cách phát hiện 2 vạch biên trắng (Camera).
- Sử dụng bộ điều khiển tối ưu LQR (Linear Quadratic Regulator) để đánh lái mượt mà.
- Tránh vật cản bằng LiDAR thông qua máy trạng thái (FSM) 3 bước dịch vạch ảo (S-Curve Ramp).

Chạy trên xe:
    python3 src/speed_track/main_speed_track.py
"""

import sys
# Sắp xếp lại sys.path để ưu tiên các thư viện Python 3 trước, tránh xung đột với ROS Python 2.7
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

import os
import rospy
import cv2
import numpy as np
import time
import math
from enum import Enum
from sensor_msgs.msg import LaserScan, Image

# Import các module điều khiển nội bộ
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.control.racer_controller import RacerController
from src.core.control.lqr_controller import LQRController, ObstacleDetector

class RobotState(Enum):
    STATE_NORMAL = 1       # Bám làn bình thường
    STATE_DODGING = 2      # Đang lách né vật cản sang phải
    STATE_REENTERING = 3   # Đang lượn quay trở lại làn cũ

class SpeedTrackController:
    def __init__(self):
        rospy.loginfo("=== KHỞI TẠO BỘ ĐIỀU KHIỂN SPEED TRACK (KHÔNG DÙNG MAP) ===")
        self.setup_parameters()
        self.initialize_hardware()

        # Trạng thái ban đầu
        self.state = RobotState.STATE_NORMAL
        self.latest_scan = None
        self.latest_image = None
        self.state_change_time = rospy.get_time()

        # Video Recorder để debug
        self.video_writer = None
        self.initialize_video_writer()

        # Đăng ký ROS Topics
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.loginfo("Đã đăng ký nhận dữ liệu từ LiDAR (/scan) và Camera (/csi_cam_0/image_raw)")

    def setup_parameters(self):
        """Cấu hình các tham số chạy xe và điều khiển (Có thể chỉnh trong docs/SPEED_TRACK_CALIBRATION.md)"""
        # 1. Tốc độ ga
        self.BASE_SPEED = 0.22         # Tốc độ di chuyển cơ bản (0.0 -> 1.0)
        
        # 2. Các tham số né tránh (LiDAR + Offset)
        self.TRIGGER_DIST = 0.70       # Khoảng cách kích hoạt né tránh (m)
        self.DODGE_OFFSET_PX = 55      # Độ rộng dịch vạch ảo để né tránh (pixel)
        self.RAMP_STEP_PX = 4          # Tốc độ dịch chuyển vạch ảo mỗi frame (pixel/frame)
        self.SIDE_CLEAR_DIST = 0.45    # Khoảng cách sườn trái an toàn trước khi nhập làn (m)

        # 3. Kích thước ảnh xử lý
        self.WIDTH = 300
        self.HEIGHT = 300

        # 4. Trạng thái dịch làn ảo hiện tại
        self.current_offset_px = 0.0
        self.target_offset_px = 0.0

        # 5. Cấu hình video ghi lại để phân tích lỗi sau lượt chạy
        self.VIDEO_OUTPUT_FILENAME = 'speed_track_run.avi'
        self.VIDEO_FPS = 20
        self.VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')

    def initialize_hardware(self):
        """Khởi tạo điều khiển phần cứng qua RacerController."""
        self.racer = RacerController()
        # Khởi tạo bộ điều khiển LQR: Chiều dài cơ sở xe ~0.18m
        self.lqr = LQRController(wheelbase=0.18, scale_factor=0.0015)
        # Khởi tạo bộ tính khoảng cách an toàn tránh vật cản
        self.detector = ObstacleDetector(reaction_time=0.5, safe_distance=0.20)
        self.racer.stop()

    def initialize_video_writer(self):
        """Khởi tạo đối tượng VideoWriter để ghi lại quá trình chạy debug."""
        try:
            self.video_writer = cv2.VideoWriter(
                self.VIDEO_OUTPUT_FILENAME,
                self.VIDEO_FOURCC,
                self.VIDEO_FPS,
                (self.WIDTH, self.HEIGHT)
            )
            rospy.loginfo(f"Bắt đầu ghi video debug vào file: '{self.VIDEO_OUTPUT_FILENAME}'")
        except Exception as e:
            rospy.logerr(f"Lỗi khởi tạo VideoWriter: {e}")
            self.video_writer = None

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def camera_callback(self, msg):
        """Chuyển đổi dữ liệu ảnh ROS byte sang mảng numpy OpenCV trực tiếp."""
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'bgr8':
                self.latest_image = img.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'rgb8':
                img_rgb = img.reshape((msg.height, msg.width, 3))
                self.latest_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                self.latest_image = img.reshape((msg.height, msg.width))
        except Exception as e:
            rospy.logerr(f"Lỗi chuyển đổi ảnh: {e}")

    # =========================================================================
    # THUẬT TOÁN XỬ LÝ ẢNH BÁM LÀN BIÊN AN TOÀN
    # =========================================================================
    def get_lane_centers(self, frame):
        """
        Phát hiện 2 vạch biên trắng và tính trung điểm đen ở giữa lòng đường.
        Tránh cán lên biên gây loại trực tiếp.
        """
        # Resize ảnh về kích thước chuẩn (300x300) để đồng bộ xử lý
        resized = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
        
        # Ngưỡng quét ROI gần và xa
        y_near = int(self.HEIGHT * 0.85)
        y_far = int(self.HEIGHT * 0.55)

        # Nhị phân lọc vạch trắng biên
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Lọc ngưỡng xám (Cân chỉnh trongdocs/SPEED_TRACK_CALIBRATION.md)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        def find_borders(y_line):
            mid_x = int(self.WIDTH / 2)
            left_border = 0
            right_border = self.WIDTH - 1

            # Quét từ giữa ảnh sang trái tìm biên trắng trái
            for x in range(mid_x, 0, -1):
                if thresh[y_line, x] == 255:
                    left_border = x
                    break
            
            # Quét từ giữa ảnh sang phải tìm biên trắng phải
            for x in range(mid_x, self.WIDTH):
                if thresh[y_line, x] == 255:
                    right_border = x
                    break
            
            center_x = int((left_border + right_border) / 2)
            return center_x, left_border, right_border

        # Lấy tọa độ tâm lòng đường của vùng gần và vùng xa
        C_near, L_near, R_near = find_borders(y_near)
        C_far, L_far, R_far = find_borders(y_far)

        # Vẽ debug trực quan lên ảnh
        debug_frame = resized.copy()
        cv2.line(debug_frame, (0, y_near), (self.WIDTH, y_near), (0, 255, 255), 1)
        cv2.line(debug_frame, (0, y_far), (self.WIDTH, y_far), (0, 255, 255), 1)
        # Vẽ biên trái/phải màu đỏ
        cv2.circle(debug_frame, (L_near, y_near), 5, (0, 0, 255), -1)
        cv2.circle(debug_frame, (R_near, y_near), 5, (0, 0, 255), -1)
        # Vẽ tâm đường bám màu xanh lá
        cv2.circle(debug_frame, (C_near, y_near), 6, (0, 255, 0), -1)
        cv2.circle(debug_frame, (C_far, y_far), 6, (0, 255, 0), -1)

        return C_near, C_far, y_near, y_far, debug_frame

    # =========================================================================
    # THUẬT TOÁN ĐO LiDAR & FSM NÉ TRÁNH
    # =========================================================================
    def get_front_obstacle_distance(self):
        """Đo khoảng cách vật cản trước mặt bằng LiDAR (-15 đến +15 độ)."""
        if self.latest_scan is None:
            return float('inf')
        
        distances = []
        msg = self.latest_scan
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            # Bù 180 độ do góc xoay lắp đặt LiDAR
            angle_deg = angle_deg + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            if -15.0 <= angle_deg <= 15.0:
                if msg.range_min < dist < msg.range_max:
                    distances.append(dist)
                    
        return min(distances) if distances else float('inf')

    def is_left_side_clear(self):
        """Đo LiDAR sườn bên trái xe (70 đến 110 độ) để check vượt qua hộp chưa."""
        if self.latest_scan is None:
            return True
        
        side_distances = []
        msg = self.latest_scan
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            angle_deg = angle_deg + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            if 70.0 <= angle_deg <= 110.0:
                if msg.range_min < dist < msg.range_max:
                    side_distances.append(dist)
                    
        if side_distances:
            return min(side_distances) > self.SIDE_CLEAR_DIST
        return True

    def update_fsm_states(self, front_dist):
        """Cập nhật máy trạng thái FSM né tránh vật cản và ramping vạch ảo."""
        
        # --- STATE 1: BÁM LÀN BÌNH THƯỜNG ---
        if self.state == RobotState.STATE_NORMAL:
            self.target_offset_px = 0.0
            
            # Kích hoạt né tránh khi khoảng cách trước mặt bé hơn ngưỡng an toàn
            if front_dist < self.TRIGGER_DIST:
                rospy.loginfo(f"⚠️ [FSM] PHÁT HIỆN VẬT CẢN: {front_dist:.2f}m. Tiến hành lách tránh!")
                self.state = RobotState.STATE_DODGING
                self.target_offset_px = self.DODGE_OFFSET_PX

        # --- STATE 2: ĐANG LÁCH NÉ VẬT CẢN (DỊCH PHẢI) ---
        elif self.state == RobotState.STATE_DODGING:
            self.target_offset_px = self.DODGE_OFFSET_PX
            
            # Check sườn trái, nếu đã thoát hoàn toàn khỏi hộp cản
            if self.is_left_side_clear():
                rospy.loginfo("✅ [FSM] Đã vượt qua vật cản. Đang lượn quay trở lại làn chính.")
                self.state = RobotState.STATE_REENTERING
                self.target_offset_px = 0.0

        # --- STATE 3: NHẬP LẠI LÀN CŨ ---
        elif self.state == RobotState.STATE_REENTERING:
            self.target_offset_px = 0.0
            
            # Khi xe thực sự đã lượn mượt mà về lại chính giữa đường
            if abs(self.current_offset_px) < 1.0:
                rospy.loginfo("🏠 [FSM] Đã về làn trung tâm thành công.")
                self.state = RobotState.STATE_NORMAL

        # --- RAMPING VẠCH ẢO (S-Curve) ---
        diff = self.target_offset_px - self.current_offset_px
        if abs(diff) > 0.1:
            step = np.sign(diff) * self.RAMP_STEP_PX
            if abs(step) > abs(diff):
                self.current_offset_px = self.target_offset_px
            else:
                self.current_offset_px += step
        else:
            self.current_offset_px = self.target_offset_px

    # =========================================================================
    # VÒNG LẶP CHẠY XE THỜI GIAN THỰC
    # =========================================================================
    def run(self):
        rate = rospy.Rate(20)  # Tần số điều khiển 20Hz (mỗi frame 50ms)
        rospy.loginfo("Bắt đầu vòng lặp điều khiển xe. Nhấn Ctrl+C để dừng khẩn cấp.")

        while not rospy.is_shutdown():
            if self.latest_image is None:
                rospy.logwarn_throttle(5, "Đang chờ ảnh từ Camera...")
                rate.sleep()
                continue

            # 1. Phát hiện đường và biên từ camera
            C_near, C_far, y_near, y_far, debug_img = self.get_lane_centers(self.latest_image)

            # 2. Tính khoảng cách trước mặt từ LiDAR
            front_dist = self.get_front_obstacle_distance()

            # 3. Cập nhật FSM và dịch làn ảo
            self.update_fsm_states(front_dist)

            # 4. Áp dụng offset né vào tâm bám đường gần
            target_center_near = C_near + self.current_offset_px
            
            # Tính sai số bẻ lái (lệch pixel giữa tâm xe 150px và tâm bám đường)
            error_px = target_center_near - (self.WIDTH / 2.0)

            # 5. Bộ điều khiển tỉ lệ Kp (Bẻ lái góc tỉ lệ với lỗi lệch)
            # Hệ số nhạy chỉnh trong docs/SPEED_TRACK_CALIBRATION.md
            Kp = 0.007
            steering = error_px * Kp

            # Giới hạn góc lái vật lý của servo lái [-1.0, 1.0]
            steering = max(-1.0, min(1.0, steering))

            # 6. Truyền lệnh điều khiển ga/lái trực tiếp xuống xe qua RacerController (I2C)
            self.racer.steer(steering, self.BASE_SPEED)

            # 7. Ghi video debug để phân tích lượt chạy
            if self.video_writer is not None:
                # Vẽ thêm thông tin FSM và khoảng cách lên ảnh để tiện phân tích video
                cv2.putText(debug_img, f"State: {self.state.name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(debug_img, f"Dist: {front_dist:.2f}m", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(debug_img, f"Offset: {self.current_offset_px:.1f}px", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                self.video_writer.write(debug_img)

            # In log debug ra màn hình Terminal
            rospy.loginfo_throttle(1, f"State: {self.state.name} | Dist: {front_dist:.2f}m | Offset: {self.current_offset_px:.1f}px | Steer: {steering:.2f}")

            rate.sleep()

        # Dừng xe khi thoát chương trình
        self.racer.stop()
        if self.video_writer is not None:
            self.video_writer.release()
            rospy.loginfo("Đã lưu video debug.")
        rospy.loginfo("Đã dừng xe an toàn.")

if __name__ == '__main__':
    try:
        controller = SpeedTrackController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"Lỗi khẩn cấp: {e}")
        # Dừng xe an toàn tuyệt đối khi lỗi phần mềm
        try:
            r = RacerController()
            r.stop()
        except:
            pass