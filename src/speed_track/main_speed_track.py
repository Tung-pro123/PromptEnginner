#!/usr/bin/env python3
"""
Jetson AI Racer Challenge 2026 - Speed Track (Bài 1)
Hệ thống điều khiển hoàn toàn tự động không sử dụng bản đồ (Mapless Autonomous).
- Bám vạch màu đỏ nét đứt (~2cm) trên sa bàn vòng tròn bằng Camera với thuật toán:
  * Biến đổi góc nhìn chim bay (IPM - Inverse Perspective Mapping)
  * Lọc màu YCrCb + CLAHE chống nhiễu bóng râm / chói sáng
  * Cửa sổ trượt (Sliding Windows) & Fit đa thức bậc 2 (Polynomial Fitting) nối liền khoảng hở nét đứt
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
    STATE_BLOCKED = 4      # Đường bị khóa 2 phía / Nguy cơ đâm -> Phanh dừng an toàn


class LaneDetector:
    """
    Bộ nhận diện vạch đường nâng cao (Computer Vision Engine):
    - Kháng ánh sáng tự nhiên bằng YCrCb + CLAHE
    - Biến đổi góc nhìn chim bay (IPM)
    - Nối liền khoảng hở vạch nét đứt bằng Sliding Windows & Polyfit Bậc 2
    """
    def __init__(self, width=300, height=300):
        self.width = width
        self.height = height

        # 1. Khởi tạo ma trận Homography biến đổi góc nhìn chim bay (IPM)
        # 4 điểm nguồn trên camera nghiêng (hình thang sa bàn)
        src_pts = np.float32([
            [15, 290],    # Dưới - Trái
            [285, 290],   # Dưới - Phải
            [205, 140],   # Trên - Phải
            [95, 140]     # Trên - Trái
        ])
        
        # 4 điểm đích trên ảnh Top-Down (hình chữ nhật chuẩn)
        dst_pts = np.float32([
            [60, 300],    # Dưới - Trái
            [240, 300],   # Dưới - Phải
            [240, 0],     # Trên - Phải
            [60, 0]       # Trên - Trái
        ])

        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

        # 2. Cân bằng tương phản cục bộ CLAHE trên kênh Y (Độ sáng)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # 3. Trạng thái đa thức và tâm đường lưu trữ (Temporal Tracking)
        self.last_coeffs = None
        self.last_c_near = 150.0
        self.last_c_far = 150.0
        self.alpha_smooth = 0.35  # Hệ số làm mượt tâm đường giữa các frame

    def transform_ipm(self, frame):
        """Chuyển đổi ảnh camera nghiêng sang góc nhìn Chim bay Top-Down chuẩn 2D"""
        return cv2.warpPerspective(frame, self.M, (self.width, self.height), flags=cv2.INTER_LINEAR)

    def extract_red_mask_ycrcb(self, warped_frame):
        """
        Tách màu vạch đỏ nét đứt bằng YCrCb + CLAHE.
        Giúp lọc ổn định bất chấp bóng râm, nhà che hay chói sáng tự nhiên.
        """
        ycrcb = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # Cân bằng tương phản kênh độ sáng Y
        y_eq = self.clahe.apply(y)

        # Lọc ngưỡng màu Đỏ trên kênh Cr (Độ chênh lệch màu Đỏ)
        # Kênh Cr cho vạch đỏ nét đứt thường có giá trị đậm trong khoảng 148 - 255
        _, mask_cr = cv2.threshold(cr, 148, 255, cv2.THRESH_BINARY)

        # Kết hợp lọc thêm ngưỡng HSV phụ để loại bỏ tuyệt đối màu trắng sáng bị lóa
        hsv = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([155, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_hsv1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_hsv2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        # Giao 2 mặt nạ YCrCb và HSV để đạt độ chính xác tối đa
        combined_mask = cv2.bitwise_and(mask_cr, mask_hsv)

        # Lọc nhiễu hạt nhỏ bằng Morphological Opening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_clean = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        return mask_clean

    def fit_polynomial_sliding_windows(self, binary_warped):
        """
        Thuật toán Cửa sổ Trượt (Sliding Windows) & Fit Đa thức Bậc 2:
        Nối liền các khoảng trống nét đứt (~2cm) và loại bỏ nhiễu điểm ngoại lai.
        """
        # 1. Tính Histogram của nửa dưới ảnh để tìm chân vạch đỏ
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        base_x = np.argmax(histogram)

        # Nếu không tìm thấy vạch đỏ trong histogram
        if histogram[base_x] < 30:
            return self.last_coeffs, False

        # 2. Cấu hình Cửa sổ Trượt (9 windows từ chân ảnh lên đỉnh)
        nwindows = 9
        window_height = binary_warped.shape[0] // nwindows
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        current_x = base_x
        margin = 45          # Độ rộng nửa cửa sổ (pixel)
        minpix = 15          # Số điểm pixel tối thiểu để cập nhật tâm cửa sổ
        lane_inds = []

        # 3. Lặp trượt cửa sổ
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_x_low = current_x - margin
            win_x_high = current_x + margin

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)

            if len(good_inds) > minpix:
                current_x = int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)

        # Nếu tổng số pixel gom được quá ít -> Không đủ tin cậy để fit đa thức mới
        if len(lane_inds) < 40:
            return self.last_coeffs, False

        x_pts = nonzerox[lane_inds]
        y_pts = nonzeroy[lane_inds]

        # 4. Nội suy đa thức bậc 2: x = a*y^2 + b*y + c
        try:
            fit_coeffs = np.polyfit(y_pts, x_pts, 2)
            self.last_coeffs = fit_coeffs
            return fit_coeffs, True
        except Exception:
            return self.last_coeffs, False

    def get_track_centers(self, frame):
        """
        Hàm xử lý chính: Chuyển IPM -> Lọc YCrCb -> Sliding Windows -> Tính C_near & C_far mượt mà
        """
        warped = self.transform_ipm(frame)
        mask = self.extract_red_mask_ycrcb(warped)
        coeffs, found = self.fit_polynomial_sliding_windows(mask)

        y_near = 230
        y_far = 150

        if coeffs is None:
            # Nếu mù vạch hoàn toàn -> Giữ nguyên vị trí tâm cũ
            return self.last_c_near, self.last_c_far, y_near, y_far, True, warped

        # Tính tọa độ X từ đa thức bậc 2: x = a*y^2 + b*y + c
        a, b, c = coeffs
        raw_c_near = a * (y_near ** 2) + b * y_near + c
        raw_c_far = a * (y_far ** 2) + b * y_far + c

        # Giới hạn tọa độ trong khoảng ảnh [0, width]
        raw_c_near = max(0.0, min(float(self.width), raw_c_near))
        raw_c_far = max(0.0, min(float(self.width), raw_c_far))

        # Làm mượt tâm đường bằng Exponential Moving Average
        c_near = self.alpha_smooth * raw_c_near + (1 - self.alpha_smooth) * self.last_c_near
        c_far = self.alpha_smooth * raw_c_far + (1 - self.alpha_smooth) * self.last_c_far

        self.last_c_near = c_near
        self.last_c_far = c_far

        # --- VẼ NỀN DEBUG ĐỂ GHI VIDEO LOG ---
        debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        ploty = np.linspace(0, self.height - 1, self.height)
        fitx = a * (ploty ** 2) + b * ploty + c
        
        # Vẽ đường cong đa thức màu xanh lá
        pts = np.asarray([np.transpose(np.vstack([fitx, ploty]))], dtype=np.int32)
        cv2.polylines(debug_img, pts, isClosed=False, color=(0, 255, 0), thickness=3)

        # Vẽ điểm tâm C_near và C_far
        cv2.circle(debug_img, (int(c_near), y_near), 6, (0, 0, 255), -1)
        cv2.circle(debug_img, (int(c_far), y_far), 6, (255, 0, 0), -1)

        return c_near, c_far, y_near, y_far, not found, debug_img


class SpeedTrackController:
    def __init__(self):
        rospy.init_node('speed_track_node', anonymous=True)
        rospy.loginfo("=== KHỞI TẠO BỘ ĐIỀU KHIỂN SPEED TRACK (IPM + YCrCb + LQR) ===")
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

        # Khởi tạo Engine Thị giác Máy tính
        self.lane_detector = LaneDetector(self.WIDTH, self.HEIGHT)

        # Video Recorder để debug
        self.video_writer = None
        self.initialize_video_writer()

        # Đăng ký ROS Topics
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        rospy.loginfo("Đã đăng ký nhận dữ liệu từ LiDAR (/scan) và Camera")

    def setup_parameters(self):
        """Cấu hình tham số di chuyển và né vật cản"""
        self.BASE_SPEED = 0.22         # Tốc độ ga cơ bản
        self.MIN_CORNER_SPEED = 0.14   # Tốc độ an toàn khi cua gắt hoặc né vật cản
        
        self.TRIGGER_DIST = 0.85       # Khoảng cách kích hoạt né né tránh (m)
        self.FRONT_SCAN_ANGLE_DEG = 20.0 # Cung quét trước mặt (±20 độ)
        self.LIDAR_MOUNT_OFFSET_DEG = 180.0 # Góc bù xoay LiDAR JetRacer (180 độ)
        
        self.DODGE_OFFSET_PX = 75      # Độ rộng dịch vạch ảo để né (pixel)
        self.RAMP_STEP_PX = 5          # Tốc độ dịch vạch ảo (pixel/frame)
        self.SIDE_CLEAR_DIST = 0.45    # Khoảng cách sườn an toàn (m)
        self.MIN_DODGE_DURATION = 1.8  # Thời gian né tối thiểu (s)
        self.MAX_DODGE_DURATION = 3.5  # Thời gian né tối đa chống kẹt (s)

        self.WIDTH = 300
        self.HEIGHT = 300

        self.current_offset_px = 0.0
        self.target_offset_px = 0.0

        self.VIDEO_OUTPUT_FILENAME = 'speed_track_run.avi'
        self.VIDEO_FPS = 20
        self.VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')

    def initialize_hardware(self):
        """Khởi tạo phần cứng RacerController & LQRController"""
        self.racer = RacerController()
        # Scale factor quy đổi 1 pixel trên ảnh IPM Top-down sang mét thực tế
        self.lqr = LQRController(wheelbase=0.18, scale_factor=0.0020)
        self.detector = ObstacleDetector(reaction_time=0.5, safe_distance=0.20)
        self.racer.stop()

    def initialize_video_writer(self):
        """Khởi tạo VideoWriter để lưu log video debug"""
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

    def get_front_obstacle_info(self):
        """
        Đo khoảng cách LiDAR trước mặt và kiểm tra độ rộng sườn an toàn (Safety Corridor Check).
        Trả về: (global_front_min, reported_obs_side, clearance_left, clearance_right, is_blocked)
        """
        if self.latest_scan is None:
            return float('inf'), 'NONE', 0.0, 0.0, False
        
        front_left_dists = []
        front_right_dists = []
        side_left_dists = []   # Góc sườn trái (+20° đến +80°)
        side_right_dists = []  # Góc sườn phải (-80° đến -20°)
        
        msg = self.latest_scan
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            angle_deg = angle_deg + self.LIDAR_MOUNT_OFFSET_DEG
            angle_deg = (angle_deg + 180) % 360 - 180
            
            if not (msg.range_min < dist < msg.range_max):
                continue

            # Cung quét trước mặt (±20 độ)
            if -self.FRONT_SCAN_ANGLE_DEG <= angle_deg <= self.FRONT_SCAN_ANGLE_DEG:
                if angle_deg >= 0.0:
                    front_left_dists.append(dist)
                else:
                    front_right_dists.append(dist)
                    
            # Cung quét sườn trái (+20° đến +80°)
            elif 20.0 < angle_deg <= 80.0:
                side_left_dists.append(dist)
                
            # Cung quét sườn phải (-80° đến -20°)
            elif -80.0 <= angle_deg < -20.0:
                side_right_dists.append(dist)

        min_front_left = min(front_left_dists) if front_left_dists else float('inf')
        min_front_right = min(front_right_dists) if front_right_dists else float('inf')
        global_front_min = min(min_front_left, min_front_right)

        if global_front_min == float('inf'):
            return float('inf'), 'NONE', float('inf'), float('inf'), False

        # Nếu thiếu tia quét sườn -> Gán 0.0 m (An toàn: Không mặc định coi là inf)
        clearance_left = min(side_left_dists) if side_left_dists else 0.0
        clearance_right = min(side_right_dists) if side_right_dists else 0.0

        SAFE_CORRIDOR = 0.45  # Khoảng trống an toàn tối thiểu sườn xe (m)

        # 1. TRƯỜNG HỢP VẬT CẢN CHÍNH GIỮA ĐƯỜNG (front_left ~ front_right)
        if abs(min_front_left - min_front_right) < 0.12:
            if clearance_right >= SAFE_CORRIDOR and clearance_right >= clearance_left:
                reported_obs_side = 'LEFT'   # Né sang PHẢI (Safe)
            elif clearance_left >= SAFE_CORRIDOR:
                reported_obs_side = 'RIGHT'  # Né sang TRÁI (Safe)
            else:
                # Cả 2 sườn đều bị khóa -> Khóa đường
                rospy.logerr("🚨 [SAFETY] Cả 2 sườn đều bị khóa! Trạng thái BLOCKED kích hoạt.")
                return global_front_min, 'BLOCKED', clearance_left, clearance_right, True

        # 2. VẬT CẢN LỆCH TRÁI -> Hướng né tự nhiên là PHẢI
        elif min_front_left < min_front_right:
            if clearance_right >= SAFE_CORRIDOR:
                reported_obs_side = 'LEFT'   # Né sang PHẢI (Safe)
            else:
                # Sườn phải hẹp/bị khóa -> Tuyệt đối KHÔNG né trái vào vật cản -> Khóa đường!
                rospy.logerr(f"🚨 [SAFETY] Vật cản bên trái, sườn phải hẹp ({clearance_right:.2f}m < {SAFE_CORRIDOR}m). DỪNG XE AN TOÀN (BLOCKED)!")
                return global_front_min, 'BLOCKED', clearance_left, clearance_right, True

        # 3. VẬT CẢN LỆCH PHẢI -> Hướng né tự nhiên là TRÁI
        else:
            if clearance_left >= SAFE_CORRIDOR:
                reported_obs_side = 'RIGHT'  # Né sang TRÁI (Safe)
            else:
                # Sườn trái hẹp/bị khóa -> Khóa đường!
                rospy.logerr(f"🚨 [SAFETY] Vật cản bên phải, sườn trái hẹp ({clearance_left:.2f}m < {SAFE_CORRIDOR}m). DỪNG XE AN TOÀN (BLOCKED)!")
                return global_front_min, 'BLOCKED', clearance_left, clearance_right, True

        return global_front_min, reported_obs_side, clearance_left, clearance_right, False

    def is_side_clear_for_reentry(self):
        """Kiểm tra khoảng cách an toàn bên sườn xe trước khi nhập lại làn chính"""
        if self.latest_scan is None:
            return False  # An toàn: Nếu mất scan thì chưa vội nhập làn ngay
            
        side_distances = []
        msg = self.latest_scan
        
        if self.dodge_direction > 0:
            angle_min_deg, angle_max_deg = 70.0, 110.0
        else:
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
        return False  # An toàn: Thiếu dữ liệu sườn không cho nhập làn vội

    def compute_dynamic_speed(self, steering):
        """
        Tự động điều chỉnh TỐC ĐỘ ĐỘNG (Dynamic Speed Scaling):
        1. Giảm ga tỷ lệ thuận với độ gắt góc đánh lái (abs(steering))
        2. Giảm ga 18% khi đang lách né/nhập làn để bám sát Checkpoint
        3. Dừng ga hẳn (0.0) khi ở trạng thái STATE_BLOCKED
        """
        if self.state == RobotState.STATE_BLOCKED:
            return 0.0

        steering_factor = max(0.0, 1.0 - 0.45 * abs(steering))
        target_speed = self.BASE_SPEED * steering_factor

        if self.state != RobotState.STATE_NORMAL:
            target_speed *= 0.82

        return max(self.MIN_CORNER_SPEED, min(self.BASE_SPEED, target_speed))

    def _set_state(self, new_state):
        rospy.loginfo(f"[STATE CHANGE] {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.state_change_time = rospy.get_time()

    def update_fsm_states(self, front_dist, obs_direction, is_blind, error_px, current_speed=0.22, is_blocked=False):
        """Cập nhật Máy trạng thái FSM né tránh vật cản & dịch làn ảo"""
        adaptive_trigger_dist = max(self.TRIGGER_DIST, self.detector.get_trigger_distance(current_speed))

        # TRƯỜNG HỢP NGUY HIỂM: ĐƯỜNG BỊ KHÓA / KHÔNG ĐỦ KHÔNG GIAN NÉ AN TOÀN
        if is_blocked or obs_direction == 'BLOCKED':
            if self.state != RobotState.STATE_BLOCKED:
                self._set_state(RobotState.STATE_BLOCKED)
            self.target_offset_px = 0.0
            return

        # STATE 4: ĐANG BỊ KHÓA ĐƯỜNG -> PHANH DỪNG VÀ CHỜ ĐƯỜNG THÔNG
        if self.state == RobotState.STATE_BLOCKED:
            self.target_offset_px = 0.0
            # Nếu vật cản đã di chuyển đi xa hoặc đường đã giải phóng an toàn -> Quay lại NORMAL
            if front_dist > adaptive_trigger_dist + 0.15 and not is_blocked:
                rospy.loginfo("✅ [SAFETY] Đường đã giải phóng an toàn! Hủy phanh dừng, quay lại NORMAL.")
                self._set_state(RobotState.STATE_NORMAL)
            return

        # STATE 1: BÁM LÀN BÌNH THƯỜNG
        if self.state == RobotState.STATE_NORMAL:
            self.target_offset_px = 0.0
            if front_dist < adaptive_trigger_dist:
                if obs_direction == 'LEFT':
                    self.dodge_direction = 1.0  # Né sang phải
                    rospy.loginfo(f"⚠️ [FSM] Né sang PHẢI (Dist: {front_dist:.2f}m < {adaptive_trigger_dist:.2f}m)")
                else:
                    self.dodge_direction = -1.0 # Né sang trái
                    rospy.loginfo(f"⚠️ [FSM] Né sang TRÁI (Dist: {front_dist:.2f}m < {adaptive_trigger_dist:.2f}m)")

                self._set_state(RobotState.STATE_DODGING)
                self.target_offset_px = self.dodge_direction * self.DODGE_OFFSET_PX

        # STATE 2: ĐANG LÁCH NÉ VẬT CẢN
        elif self.state == RobotState.STATE_DODGING:
            self.target_offset_px = self.dodge_direction * self.DODGE_OFFSET_PX
            dodge_duration = rospy.get_time() - self.state_change_time
            
            if dodge_duration > self.MAX_DODGE_DURATION:
                rospy.logwarn("⚠️ [FSM] Quá thời gian né tránh tối đa. Nhập lại làn!")
                self._set_state(RobotState.STATE_REENTERING)
                self.target_offset_px = 0.0
            elif dodge_duration > self.MIN_DODGE_DURATION:
                if self.is_side_clear_for_reentry():
                    rospy.loginfo("✅ [FSM] Sườn đã trống. Đang quay trở lại làn chính.")
                    self._set_state(RobotState.STATE_REENTERING)
                    self.target_offset_px = 0.0

        # STATE 3: NHẬP LẠI LÀN CŨ
        elif self.state == RobotState.STATE_REENTERING:
            self.target_offset_px = 0.0
            reenter_duration = rospy.get_time() - self.state_change_time
            
            if abs(self.current_offset_px) < 1.0:
                if (not is_blind and abs(error_px) < 25.0) or reenter_duration > 2.5:
                    rospy.loginfo("🏠 [FSM] Về làn trung tâm thành công.")
                    self._set_state(RobotState.STATE_NORMAL)

        # RAMPING VẠCH ẢO S-CURVE
        diff = self.target_offset_px - self.current_offset_px
        if abs(diff) > 0.1:
            step = np.sign(diff) * self.RAMP_STEP_PX
            if abs(step) > abs(diff):
                self.current_offset_px = self.target_offset_px
            else:
                self.current_offset_px += step
        else:
            self.current_offset_px = self.target_offset_px

    def run(self):
        rate = rospy.Rate(20)
        rospy.loginfo("Bắt đầu vòng lặp điều khiển xe với IPM + Polyfit LQR + Smart Avoidance. Nhấn Ctrl+C để dừng khẩn cấp.")

        while not rospy.is_shutdown():
            now = rospy.get_time()

            # Timeout bảo vệ mất tín hiệu cảm biến
            if self.latest_image is not None and (now - self.last_image_time > 0.6):
                rospy.logerr_throttle(2, "🚨 [MẤT TÍN HIỆU CAMERA] Dừng xe khẩn cấp.")
                self.racer.stop()
                rate.sleep()
                continue

            if self.latest_scan is not None and (now - self.last_scan_time > 0.6):
                rospy.logerr_throttle(2, "🚨 [MẤT TÍN HIỆU LIDAR] Dừng xe khẩn cấp.")
                self.racer.stop()
                rate.sleep()
                continue

            if self.latest_image is None:
                rospy.logwarn_throttle(5, "Đang chờ ảnh từ Camera...")
                rate.sleep()
                continue

            # 1. Nhận diện vạch đường mượt mà bằng LaneDetector (IPM + Polyfit)
            C_near, C_far, y_near, y_far, is_blind, debug_img = self.lane_detector.get_track_centers(self.latest_image)

            # 2. Đo khoảng cách vật cản & Kiểm tra sườn trống thông minh (Safety Corridor Check)
            front_dist, obs_direction, clearance_left, clearance_right, is_blocked = self.get_front_obstacle_info()

            temp_error_px = C_near - (self.WIDTH / 2.0)

            # 3. Cập nhật FSM & Dịch vạch ảo né tránh (Xử lý trạng thái BLOCKED an toàn)
            self.update_fsm_states(front_dist, obs_direction, is_blind, temp_error_px, current_speed=self.BASE_SPEED, is_blocked=is_blocked)

            # 4. Tính toán sai số bẻ lái
            target_center_near = C_near + self.current_offset_px
            error_px = target_center_near - (self.WIDTH / 2.0)

            # 5. Tự động tính toán tốc độ động thực tế (Dynamic Speed Scaling)
            # Ước tính góc lái tạm thời để lấy tốc độ động cho LQR Riccati Matrix
            raw_err_far = (C_far - C_near)
            est_steering_preview = np.clip(0.005 * error_px + 0.003 * raw_err_far, -1.0, 1.0)
            current_speed = self.compute_dynamic_speed(est_steering_preview)

            # 6. Đồng bộ LQR Offset & Tính toán góc lái mượt mà với tốc độ động thực tế
            self.lqr.target_offset = self.target_offset_px * self.lqr.scale_factor
            self.lqr.current_offset = self.current_offset_px * self.lqr.scale_factor
            
            steering = self.lqr.compute_steering(
                C_near=C_near,
                C_far=C_far,
                Y_near=y_near,
                Y_far=y_far,
                speed=current_speed,
                image_width=self.WIDTH
            )

            # Fallback hỗ trợ né tránh nếu mất vạch tạm thời
            if is_blind and self.state == RobotState.STATE_REENTERING:
                steering = -0.35 * self.dodge_direction

            steering = max(-1.0, min(1.0, steering))

            # 7. Cập nhật tốc độ chính thức và truyền lệnh điều khiển xuống RacerController (I2C)
            current_speed = self.compute_dynamic_speed(steering)
            self.racer.steer(steering, current_speed)

            # 8. Ghi video debug với giao diện theo dõi trực quan
            if self.video_writer is not None:
                cv2.putText(debug_img, f"State: {self.state.name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(debug_img, f"Dist: {front_dist:.2f}m | Speed: {current_speed:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(debug_img, f"Offset: {self.current_offset_px:.1f}px", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                if is_blind:
                    cv2.putText(debug_img, "BLIND - TEMPORAL FALLBACK", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                self.video_writer.write(debug_img)

            # In log debug ra màn hình Terminal
            if is_blind:
                rospy.logwarn_throttle(1, f"🚨 [TEMPORAL FALLBACK] Center: {C_near:.1f} | Error: {error_px:.1f}px | Steer: {steering:.2f} | Speed: {current_speed:.2f}")
            else:
                rospy.loginfo_throttle(1, f"State: {self.state.name} | Error: {error_px:.1f}px | Steer: {steering:.2f} | Speed: {current_speed:.2f} | Dist: {front_dist:.2f}m")

            rate.sleep()

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
        try:
            r = RacerController()
            r.stop()
        except:
            pass