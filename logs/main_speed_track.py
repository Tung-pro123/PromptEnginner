#!/usr/bin/env python3
"""
Jetson AI Racer Challenge 2026 - Speed Track (Bài 1)
Hệ thống điều khiển hoàn toàn tự động không sử dụng bản đồ (Mapless Autonomous).
- Bám vạch màu đỏ nét đứt (~2cm) trên sa bàn vòng tròn bằng Camera (HSV Dual-Threshold).
- Sử dụng bộ điều khiển tối ưu LQR (Linear Quadratic Regulator) bám đuổi tâm đường & góc cua.
- Tránh vật cản bằng LiDAR thông qua máy trạng thái (FSM) 3 bước dịch vạch ảo mượt mà (S-Curve Ramp).
- Tự động điều chỉnh tốc độ mượt mà khi vào cua gắt để bắt đúng các Checkpoint điểm số.

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
    STATE_DODGING = 2      # Đang lách né vật cản
    STATE_REENTERING = 3   # Đang lượn quay trở lại làn cũ

class SpeedTrackController:
    def __init__(self):
        rospy.init_node('speed_track_node', anonymous=True)
        rospy.loginfo("=== KHỞI TẠO BỘ ĐIỀU KHIỂN SPEED TRACK (VẠCH ĐỎ NÉT ĐỨT & VẬT CẢN) ===")
        self.setup_parameters()
        self.initialize_hardware()

        # Trạng thái ban đầu
        self.state = RobotState.STATE_NORMAL
        self.latest_scan = None
        self.latest_image = None
        self.last_image_time = rospy.get_time()
        self.last_scan_time = rospy.get_time()
        self.state_change_time = rospy.get_time()
        self.dodge_direction = 1.0  # 1.0 = Tránh phải, -1.0 = Tránh trái

        # Bộ nhớ tâm đường để bám đuổi (Temporal Tracking)
        self.last_C_near = 150.0
        self.last_C_far = 150.0

        # Video Recorder để debug
        self.video_writer = None
        self.initialize_video_writer()

        # Đăng ký ROS Topics
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        rospy.loginfo("Đã đăng ký nhận dữ liệu từ LiDAR (/scan) và Camera")

    def setup_parameters(self):
        """Cấu hình các tham số chạy xe và điều khiển (Có thể chỉnh trong docs/SPEED_TRACK_CALIBRATION.md)"""
        # 1. Tốc độ ga cơ bản & Tốc độ tối thiểu khi cua/né
        self.BASE_SPEED = 0.22         # Tốc độ di chuyển cơ bản (0.0 -> 1.0)
        self.MIN_CORNER_SPEED = 0.14   # Tốc độ an toàn tối thiểu khi cua gắt hoặc né vật cản
        
        # 2. Các tham số né tránh (LiDAR + Offset)
        self.TRIGGER_DIST = 0.85       # Khoảng cách kích hoạt né tránh (m)
        self.FRONT_SCAN_ANGLE_DEG = 20.0 # Mở rộng góc quét trước mặt (±20 độ) để phát hiện sớm trên cua
        self.LIDAR_MOUNT_OFFSET_DEG = 180.0 # Góc xoay lắp đặt LiDAR trên xe JetRacer (180 độ)
        
        self.DODGE_OFFSET_PX = 75      # Độ rộng dịch vạch ảo để né tránh (pixel) (tương đương ~11.2cm thực tế)
        self.RAMP_STEP_PX = 5          # Tốc độ dịch chuyển vạch ảo mỗi frame (pixel/frame)
        self.SIDE_CLEAR_DIST = 0.45    # Khoảng cách sườn an toàn trước khi nhập làn (m)
        self.MIN_DODGE_DURATION = 1.8  # Thời gian tối thiểu né tránh vật cản trước khi nhập làn (s)
        self.MAX_DODGE_DURATION = 3.5  # Thời gian tối đa né tránh để tự động nhập làn nếu bị kẹt cảm biến (s)

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

        # 6. Các tham số lọc hình học vạch đứt và bộ nhớ tạm thời (Tối ưu cho vạch đỏ nét đứt ~2cm)
        self.MIN_LANE_AREA = 10        # Diện tích tối thiểu để lọc vạch 2cm (pixel)
        self.MAX_LANE_AREA = 1500      # Diện tích tối đa để loại bỏ hộp cản/bức tường (pixel)
        self.MIN_ASPECT_RATIO = 1.1    # Tỷ lệ dài/rộng tối thiểu cho nét đứt 2cm
        self.SMOOTHING_ALPHA = 0.30    # Hệ số làm mượt tâm đường (0.0 -> 1.0)

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
        self.last_scan_time = rospy.get_time()

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
            self.last_image_time = rospy.get_time()
        except Exception as e:
            rospy.logerr(f"Lỗi chuyển đổi ảnh: {e}")

    # =========================================================================
    # THUẬT TOÁN XỬ LÝ ẢNH BÁM LÀN BIÊN AN TOÀN (LỌC MÀU ĐỎ NÉT ĐỨT 2CM)
    # =========================================================================
    def get_lane_centers(self, frame):
        """
        Phát hiện vạch đỏ nét đứt (~2cm) bằng cách chuyển không gian màu HSV, 
        lọc kép dải Hue đỏ và tìm Contours hình học.
        Bảo vệ xe không bị mất vạch khi đi qua khe hở nét đứt bằng bộ nhớ Temporal Tracking.
        """
        # Resize ảnh về kích thước chuẩn (300x300) để đồng bộ xử lý
        resized = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
        
        # Xác định dải ROI quét cho vùng gần và xa
        y_near = int(self.HEIGHT * 0.73)
        h_near = int(self.HEIGHT * 0.12)  # Quét dải cao 36 pixel vùng gần
        
        y_far = int(self.HEIGHT * 0.50)
        h_far = int(self.HEIGHT * 0.12)   # Quét dải cao 36 pixel vùng xa

        # Chuyển đổi ảnh sang không gian màu HSV để lọc màu ĐỎ chính xác
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Màu đỏ trong HSV nằm ở 2 dải H (0-10 và 160-180)
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        thresh = cv2.bitwise_or(mask1, mask2)

        # Lọc nhiễu hạt bằng Morphological Opening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # --- Áp dụng che nhiễu vật cản (Spatial Masking) ---
        if self.state == RobotState.STATE_DODGING:
            if self.dodge_direction > 0:
                thresh[:, 0:int(self.WIDTH / 2)] = 0
            else:
                thresh[:, int(self.WIDTH / 2):] = 0

        # Hàm tìm trọng tâm vạch trong một dải quét dọc
        def find_lane_centroid_in_roi(y_start, height, last_C):
            roi = thresh[y_start : y_start + height, :]
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_x = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Lọc theo diện tích vạch
                if self.MIN_LANE_AREA < area < self.MAX_LANE_AREA:
                    # Lấy hình chữ nhật bao quanh nhỏ nhất để check tỉ lệ thuôn dài
                    rect = cv2.minAreaRect(cnt)
                    (x, y), (w, h), angle = rect
                    longer_side = max(w, h)
                    shorter_side = min(w, h) + 1e-5
                    aspect_ratio = longer_side / shorter_side
                    
                    # Vạch nét đứt hướng tâm thường thuôn dài hình chữ nhật
                    if aspect_ratio > self.MIN_ASPECT_RATIO:
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            valid_x.append(cx)
            
            if valid_x:
                # Chọn vạch gần với vị trí tâm của khung hình trước nhất để tránh nhảy vạch đột ngột
                closest_cx = min(valid_x, key=lambda val: abs(val - last_C))
                # Lọc thông thấp mượt mà
                new_C = self.SMOOTHING_ALPHA * closest_cx + (1 - self.SMOOTHING_ALPHA) * last_C
                return new_C, True
            
            # Gặp khoảng trống (khe hở nét đứt): giữ nguyên vị trí cũ
            return last_C, False

        # Tìm tâm đường vùng gần và xa
        C_near, found_near = find_lane_centroid_in_roi(y_near, h_near, self.last_C_near)
        C_far, found_far = find_lane_centroid_in_roi(y_far, h_far, self.last_C_far)

        # Cập nhật bộ nhớ tâm đường
        self.last_C_near = C_near
        self.last_C_far = C_far

        # Xác định trạng thái xem có phát hiện vạch thực tế không (is_blind)
        is_blind = not (found_near or found_far)

        # --- VẼ DEBUG TRỰC QUAN ĐỂ GHI VIDEO ---
        debug_frame = resized.copy()
        # Vẽ các vùng quét ROI bằng màu vàng
        cv2.rectangle(debug_frame, (0, y_near), (self.WIDTH, y_near + h_near), (0, 255, 255), 1)
        cv2.rectangle(debug_frame, (0, y_far), (self.WIDTH, y_far + h_far), (0, 255, 255), 1)
        
        # Vẽ tâm đường bám đuổi bằng màu xanh lá
        cv2.circle(debug_frame, (int(C_near), y_near + int(h_near/2)), 6, (0, 255, 0), -1)
        cv2.circle(debug_frame, (int(C_far), y_far + int(h_far/2)), 6, (0, 255, 0), -1)

        return C_near, C_far, y_near + int(h_near/2), y_far + int(h_far/2), is_blind, debug_frame

    # =========================================================================
    # THUẬT TOÁN ĐO LiDAR & FSM NÉ TRÁNH
    # =========================================================================
    def get_front_obstacle_info(self):
        """
        Đo khoảng cách vật cản trước mặt trong cung ±FRONT_SCAN_ANGLE_DEG và trả về (min_dist, direction)
        direction: 'LEFT' nếu vật cản lệch trái (ta cần tránh sang phải),
                   'RIGHT' nếu vật cản lệch phải (ta cần tránh sang trái).
        """
        if self.latest_scan is None:
            return float('inf'), 'LEFT'
        
        left_dists = []
        right_dists = []
        msg = self.latest_scan
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            # Bù góc 180 độ do LiDAR xoay ngược trên xe JetRacer
            angle_deg = angle_deg + self.LIDAR_MOUNT_OFFSET_DEG
            angle_deg = (angle_deg + 180) % 360 - 180
            
            # Cung quét trước mặt từ -FRONT_SCAN_ANGLE_DEG đến +FRONT_SCAN_ANGLE_DEG độ
            if -self.FRONT_SCAN_ANGLE_DEG <= angle_deg <= self.FRONT_SCAN_ANGLE_DEG:
                if msg.range_min < dist < msg.range_max:
                    if angle_deg >= 0.0:  # Trái trong hệ tọa độ ROS (+Y left)
                        left_dists.append(dist)
                    else:                 # Phải trong hệ tọa độ ROS (-Y right)
                        right_dists.append(dist)
                        
        min_left = min(left_dists) if left_dists else float('inf')
        min_right = min(right_dists) if right_dists else float('inf')
        
        global_min = min(min_left, min_right)
        if global_min == float('inf'):
            return float('inf'), 'LEFT'
            
        # Nếu khoảng cách bên trái ngắn hơn -> Vật cản lệch trái -> Ta né sang PHẢI
        # Ngược lại -> Vật cản lệch phải -> Ta né sang TRÁI
        if min_left < min_right:
            return global_min, 'LEFT'
        else:
            return global_min, 'RIGHT'

    def is_side_clear_for_reentry(self):
        """
        Kiểm tra xem sườn xe đối diện với hướng né đã trống hoàn toàn chưa.
        - Nếu ta đang né sang phải (dodge_direction = 1.0) -> Check sườn TRÁI (+70 đến +110 độ)
        - Nếu ta đang né sang trái (dodge_direction = -1.0) -> Check sườn PHẢI (-110 đến -70 độ)
        """
        if self.latest_scan is None:
            return True
            
        side_distances = []
        msg = self.latest_scan
        
        # Xác định cung quét sườn theo hướng né
        if self.dodge_direction > 0:
            # Né phải -> Check sườn trái (+70 đến +110 deg)
            angle_min_deg, angle_max_deg = 70.0, 110.0
        else:
            # Né trái -> Check sườn phải (-110 đến -70 deg)
            angle_min_deg, angle_max_deg = -110.0, -70.0
            
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            angle_deg = angle_deg + self.LIDAR_MOUNT_OFFSET_DEG
            angle_deg = (angle_deg + 180) % 360 - 180
            
            if angle_min_deg <= angle_deg <= angle_max_deg:
                if msg.range_min < dist < msg.range_max:
                    side_distances.append(dist)
                    
        if side_distances:
            return min(side_distances) > self.SIDE_CLEAR_DIST
        return True

    def _set_state(self, new_state):
        rospy.loginfo(f"[STATE CHANGE] {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.state_change_time = rospy.get_time()

    def update_fsm_states(self, front_dist, obs_direction, is_blind, error_px, current_speed=0.22):
        """Cập nhật máy trạng thái FSM né tránh vật cản và ramping vạch ảo."""
        
        # Ngưỡng kích hoạt né tránh thích ứng với tốc độ xe qua ObstacleDetector
        adaptive_trigger_dist = max(self.TRIGGER_DIST, self.detector.get_trigger_distance(current_speed))

        # --- STATE 1: BÁM LÀN BÌNH THƯỜNG ---
        if self.state == RobotState.STATE_NORMAL:
            self.target_offset_px = 0.0
            
            # Kích hoạt né tránh khi khoảng cách trước mặt bé hơn ngưỡng an toàn thích ứng
            if front_dist < adaptive_trigger_dist:
                if obs_direction == 'LEFT':
                    self.dodge_direction = 1.0  # Vật cản lệch trái -> Tránh sang phải (offset > 0)
                    rospy.loginfo(f"⚠️ [FSM] Vật cản LỆCH TRÁI ({front_dist:.2f}m < {adaptive_trigger_dist:.2f}m). Né sang PHẢI!")
                else:
                    self.dodge_direction = -1.0  # Vật cản lệch phải -> Tránh sang trái (offset < 0)
                    rospy.loginfo(f"⚠️ [FSM] Vật cản LỆCH PHẢI ({front_dist:.2f}m < {adaptive_trigger_dist:.2f}m). Né sang TRÁI!")

                self._set_state(RobotState.STATE_DODGING)
                self.target_offset_px = self.dodge_direction * self.DODGE_OFFSET_PX

        # --- STATE 2: ĐANG LÁCH NÉ VẬT CẢN (DỊCH PHẢI/TRÁI) ---
        elif self.state == RobotState.STATE_DODGING:
            self.target_offset_px = self.dodge_direction * self.DODGE_OFFSET_PX
            
            # Tính thời gian đã ở trong trạng thái né tránh
            dodge_duration = rospy.get_time() - self.state_change_time
            
            # Điều kiện 1: Đạt thời gian tối đa (Bảo vệ chống kẹt do đọc nhầm tường/người đứng bên hông)
            if dodge_duration > self.MAX_DODGE_DURATION:
                rospy.logwarn("⚠️ [FSM] Quá thời gian né tránh tối đa. Tự động chuyển làn nhập lại để tránh đâm tường!")
                self._set_state(RobotState.STATE_REENTERING)
                self.target_offset_px = 0.0
            # Điều kiện 2: Vượt quá thời gian tối thiểu VÀ sườn đối diện thực sự trống
            elif dodge_duration > self.MIN_DODGE_DURATION:
                # Check sườn xem đã thoát hoàn toàn khỏi hộp cản chưa
                if self.is_side_clear_for_reentry():
                    rospy.loginfo("✅ [FSM] Đã vượt qua vật cản. Đang lượn quay trở lại làn chính.")
                    self._set_state(RobotState.STATE_REENTERING)
                    self.target_offset_px = 0.0

        # --- STATE 3: NHẬP LẠI LÀN CŨ ---
        elif self.state == RobotState.STATE_REENTERING:
            self.target_offset_px = 0.0
            
            reenter_duration = rospy.get_time() - self.state_change_time
            
            # Điều kiện chuyển về NORMAL:
            # 1. Dịch vạch ảo đã về 0
            # 2. Đã nhìn thấy vạch sa bàn thực tế và xe đã tương đối nằm giữa làn (error_px < 25px)
            # HOẶC Quá thời gian nhập làn tối đa (2.5 giây) làm điều kiện thoát an toàn
            if abs(self.current_offset_px) < 1.0:
                if (not is_blind and abs(error_px) < 25.0) or reenter_duration > 2.5:
                    rospy.loginfo("🏠 [FSM] Đã về làn trung tâm thành công. Quay lại chế độ bám làn bình thường.")
                    self._set_state(RobotState.STATE_NORMAL)

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
            now = rospy.get_time()

            # Kiểm tra dữ liệu cảm biến bị đứng hình (Stale data timeout > 0.6s)
            if self.latest_image is not None and (now - self.last_image_time > 0.6):
                rospy.logerr_throttle(2, "🚨 [MẤT TÍN HIỆU CAMERA] Dữ liệu ảnh quá 0.6s! Dừng xe khẩn cấp.")
                self.racer.stop()
                rate.sleep()
                continue

            if self.latest_scan is not None and (now - self.last_scan_time > 0.6):
                rospy.logerr_throttle(2, "🚨 [MẤT TÍN HIỆU LIDAR] Dữ liệu scan quá 0.6s! Dừng xe khẩn cấp.")
                self.racer.stop()
                rate.sleep()
                continue

            if self.latest_image is None:
                rospy.logwarn_throttle(5, "Đang chờ ảnh từ Camera...")
                rate.sleep()
                continue

            # 1. Phát hiện đường và biên từ camera
            C_near, C_far, y_near, y_far, is_blind, debug_img = self.get_lane_centers(self.latest_image)

            # 2. Tính khoảng cách trước mặt từ LiDAR và xác định hướng vật cản
            front_dist, obs_direction = self.get_front_obstacle_info()

            # Tính sai số bẻ lái tạm thời để FSM kiểm tra mức độ về làn thực tế
            temp_error_px = C_near - (self.WIDTH / 2.0)

            # 3. Cập nhật FSM và dịch làn ảo (Sử dụng ngưỡng né thích ứng tốc độ)
            self.update_fsm_states(front_dist, obs_direction, is_blind, temp_error_px, current_speed=self.BASE_SPEED)

            # 4. Áp dụng offset né vào tâm bám đường gần
            target_center_near = C_near + self.current_offset_px
            
            # Tính sai số bẻ lái chính thức (bao gồm cả offset vạch ảo)
            error_px = target_center_near - (self.WIDTH / 2.0)

            # 5. Bộ điều khiển tối ưu LQR (Đồng bộ hóa Offset vạch ảo chính xác quy đổi từ pixel sang mét)
            self.lqr.target_offset = self.target_offset_px * self.lqr.scale_factor
            self.lqr.current_offset = self.current_offset_px * self.lqr.scale_factor
            
            # Tính toán góc lái mượt mà bằng LQR
            # Note: Do lqr.current_offset đã được cập nhật đồng bộ, ta truyền C_near gốc để LQR tính sai số
            steering = self.lqr.compute_steering(
                C_near=C_near,
                C_far=C_far,
                Y_near=y_near,
                Y_far=y_far,
                speed=self.BASE_SPEED,
                image_width=self.WIDTH
            )

            # Fallback nếu xe bị mất vạch khi đang lách tránh/nhập làn: 
            # Ép xe cua nhẹ ngược lại hướng né cũ để tìm lại làn đường chính thay vì chạy thẳng tuột ra ngoài
            if is_blind and self.state == RobotState.STATE_REENTERING:
                steering = -0.35 * self.dodge_direction  # Bẻ lái ngược hướng né để quay đầu về phía sa bàn

            # Giới hạn góc lái vật lý của servo lái [-1.0, 1.0]
            steering = max(-1.0, min(1.0, steering))

            # 6. Điều chỉnh tốc độ linh hoạt (Dynamic Speed Scaling)
            # Giảm tốc nhẹ khi đang né/nhập làn hoặc khi bẻ lái góc gắt để bánh bám đường tốt qua Checkpoint
            if self.state != RobotState.STATE_NORMAL or abs(steering) > 0.35:
                current_speed = max(self.MIN_CORNER_SPEED, self.BASE_SPEED * (1.0 - 0.35 * abs(steering)))
            else:
                current_speed = self.BASE_SPEED

            # 7. Truyền lệnh điều khiển ga/lái trực tiếp xuống xe qua RacerController (I2C)
            self.racer.steer(steering, current_speed)

            # 8. Ghi video debug để phân tích lượt chạy
            if self.video_writer is not None:
                # Vẽ thêm thông tin FSM và khoảng cách lên ảnh để tiện phân tích video
                cv2.putText(debug_img, f"State: {self.state.name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(debug_img, f"Dist: {front_dist:.2f}m | Speed: {current_speed:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(debug_img, f"Offset: {self.current_offset_px:.1f}px", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                if is_blind:
                    cv2.putText(debug_img, "BLIND - NO LINES!", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                self.video_writer.write(debug_img)

            # In log debug ra màn hình Terminal
            if is_blind:
                rospy.logwarn_throttle(1, f"🚨 [MÙ VẠCH] Center: {C_near:.1f} | Error: {error_px:.1f}px | Steer: {steering:.2f} | Speed: {current_speed:.2f} | Dist: {front_dist:.2f}m")
            else:
                rospy.loginfo_throttle(1, f"State: {self.state.name} | Error: {error_px:.1f}px | Steer: {steering:.2f} | Speed: {current_speed:.2f} | Dist: {front_dist:.2f}m")

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