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
        rospy.init_node('speed_track_node', anonymous=True)
        rospy.loginfo("=== KHỞI TẠO BỘ ĐIỀU KHIỂN SPEED TRACK (KHÔNG DÙNG MAP) ===")
        self.setup_parameters()
        self.initialize_hardware()

        # Trạng thái ban đầu
        self.state = RobotState.STATE_NORMAL
        self.latest_scan = None
        self.latest_image = None
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
        # 1. Tốc độ ga
        self.BASE_SPEED = 0.22         # Tốc độ di chuyển cơ bản (0.0 -> 1.0)
        
        # 2. Các tham số né tránh (LiDAR + Offset)
        self.TRIGGER_DIST = 0.85       # Khoảng cách kích hoạt né tránh (m)
        self.DODGE_OFFSET_PX = 110     # Độ rộng dịch vạch ảo để né tránh (pixel) (tương đương ~16.5cm)
        self.RAMP_STEP_PX = 10         # Tốc độ dịch chuyển vạch ảo mỗi frame (pixel/frame)
        self.SIDE_CLEAR_DIST = 0.45    # Khoảng cách sườn trái an toàn trước khi nhập làn (m)
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
        Sử dụng bộ nhớ tâm làn để quét từ trong ra ngoài một cách linh hoạt (Temporal Tracking).
        """
        # Resize ảnh về kích thước chuẩn (300x300) để đồng bộ xử lý
        resized = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
        
        # Ngưỡng quét ROI gần và xa (Nâng cao lên để tránh quét đè lên cản trước màu xanh lá của xe)
        y_near = int(self.HEIGHT * 0.73)
        y_far = int(self.HEIGHT * 0.50)

        # Nhị phân lọc vạch trắng biên
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Lọc ngưỡng xám động (Adaptive Thresholding bằng phân vị để thích ứng ánh sáng thay đổi)
        dynamic_thresh_val = max(110, min(220, int(np.percentile(gray, 92))))
        _, thresh = cv2.threshold(gray, dynamic_thresh_val, 255, cv2.THRESH_BINARY)

        # Lấy điểm xuất phát quét từ bộ nhớ tâm đường của khung hình trước
        start_near = int(self.last_C_near)
        start_far = int(self.last_C_far)

        # Đảm bảo điểm xuất phát nằm trong giới hạn an toàn của ảnh
        start_near = max(10, min(self.WIDTH - 11, start_near))
        start_far = max(10, min(self.WIDTH - 11, start_far))

        def find_borders(y_line, start_x):
            left_border = 0
            right_border = self.WIDTH - 1

            # Quét từ điểm xuất phát sang trái tìm biên trắng trái
            for x in range(start_x, 0, -1):
                if thresh[y_line, x] == 255:
                    left_border = x
                    break
            
            # Quét từ điểm xuất phát sang phải tìm biên trắng phải
            for x in range(start_x, self.WIDTH):
                if thresh[y_line, x] == 255:
                    right_border = x
                    break
            
            center_x = int((left_border + right_border) / 2)
            return center_x, left_border, right_border

        # Lấy tọa độ tâm lòng đường của vùng gần và vùng xa
        C_near, L_near, R_near = find_borders(y_near, start_near)
        C_far, L_far, R_far = find_borders(y_far, start_far)

        # Cập nhật bộ nhớ tâm đường nếu tìm thấy vạch hợp lệ (tránh cập nhật khi bị mù vạch)
        if L_near > 0 or R_near < self.WIDTH - 1:
            self.last_C_near = C_near
        else:
            # Fallback về giữa nếu mất vạch quá lâu
            self.last_C_near = 150.0

        if L_far > 0 or R_far < self.WIDTH - 1:
            self.last_C_far = C_far
        else:
            self.last_C_far = 150.0

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

        return C_near, C_far, y_near, y_far, L_near, R_near, debug_frame

    # =========================================================================
    # THUẬT TOÁN ĐO LiDAR & FSM NÉ TRÁNH
    # =========================================================================
    def get_front_obstacle_info(self):
        """
        Đo khoảng cách vật cản trước mặt và trả về (min_dist, direction)
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
            angle_deg = angle_deg + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            # Cung quét trước mặt từ -15.0 đến 15.0 độ
            if -15.0 <= angle_deg <= 15.0:
                if msg.range_min < dist < msg.range_max:
                    if angle_deg >= 0.0:
                        left_dists.append(dist)
                    else:
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
        - Nếu ta đang né sang phải (dodge_direction = 1.0) -> Check sườn TRÁI (70 đến 110 độ)
        - Nếu ta đang né sang trái (dodge_direction = -1.0) -> Check sườn PHẢI (-110 đến -70 độ)
        """
        if self.latest_scan is None:
            return True
            
        side_distances = []
        msg = self.latest_scan
        
        # Xác định cung quét sườn theo hướng né
        if self.dodge_direction > 0:
            # Né phải -> Check sườn trái
            angle_min_deg, angle_max_deg = 70.0, 110.0
        else:
            # Né trái -> Check sườn phải
            angle_min_deg, angle_max_deg = -110.0, -70.0
            
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            angle_deg = angle_deg + 180.0
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

    def update_fsm_states(self, front_dist, obs_direction, is_blind, error_px):
        """Cập nhật máy trạng thái FSM né tránh vật cản và ramping vạch ảo."""
        
        # --- STATE 1: BÁM LÀN BÌNH THƯỜNG ---
        if self.state == RobotState.STATE_NORMAL:
            self.target_offset_px = 0.0
            
            # Kích hoạt né tránh khi khoảng cách trước mặt bé hơn ngưỡng an toàn
            if front_dist < self.TRIGGER_DIST:
                if obs_direction == 'LEFT':
                    self.dodge_direction = 1.0  # Vật cản lệch trái -> Tránh sang phải (offset > 0)
                    rospy.loginfo(f"⚠️ [FSM] Vật cản LỆCH TRÁI ({front_dist:.2f}m). Né sang PHẢI!")
                else:
                    self.dodge_direction = -1.0  # Vật cản lệch phải -> Tránh sang trái (offset < 0)
                    rospy.loginfo(f"⚠️ [FSM] Vật cản LỆCH PHẢI ({front_dist:.2f}m). Né sang TRÁI!")

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
            # HOẶC Quá thời gian nhập làn tối đa (2.5 giây) làm điều kiện thoát an toàn (tránh bị kẹt lái trái)
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
            if self.latest_image is None:
                rospy.logwarn_throttle(5, "Đang chờ ảnh từ Camera...")
                rate.sleep()
                continue

            # 1. Phát hiện đường và biên từ camera
            C_near, C_far, y_near, y_far, L_near, R_near, debug_img = self.get_lane_centers(self.latest_image)

            # Kiểm tra xem có nhận dạng được vạch biên không
            is_blind = (L_near == 0 and R_near == self.WIDTH - 1)

            # 2. Tính khoảng cách trước mặt từ LiDAR và xác định hướng vật cản
            front_dist, obs_direction = self.get_front_obstacle_info()

            # Tính sai số bẻ lái tạm thời để FSM kiểm tra mức độ về làn thực tế
            temp_error_px = C_near - (self.WIDTH / 2.0)

            # 3. Cập nhật FSM và dịch làn ảo
            self.update_fsm_states(front_dist, obs_direction, is_blind, temp_error_px)

            # 4. Áp dụng offset né vào tâm bám đường gần
            target_center_near = C_near + self.current_offset_px
            
            # Tính sai số bẻ lái chính thức (bao gồm cả offset vạch ảo)
            error_px = target_center_near - (self.WIDTH / 2.0)

            # 5. Bộ điều khiển tỉ lệ Kp (Bẻ lái góc tỉ lệ với lỗi lệch)
            # Hệ số nhạy chỉnh trong docs/SPEED_TRACK_CALIBRATION.md
            Kp = 0.015
            steering = error_px * Kp

            # Fallback nếu xe bị mất vạch khi đang lách tránh/nhập làn: 
            # Ép xe cua nhẹ ngược lại hướng né cũ để tìm lại làn đường chính thay vì chạy thẳng tuột ra ngoài
            if is_blind and self.state == RobotState.STATE_REENTERING:
                steering = -0.35 * self.dodge_direction  # Bẻ lái ngược hướng né để quay đầu về phía sa bàn

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
                if is_blind:
                    cv2.putText(debug_img, "BLIND - NO LINES!", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                self.video_writer.write(debug_img)

            # In log debug ra màn hình Terminal kèm cảnh báo nếu camera bị mù vạch
            if is_blind:
                rospy.logwarn_throttle(1, f"🚨 [CẢNH BÁO: MÙ VẠCH] L/R: {L_near}/{R_near} | Center: {C_near} | Error: {error_px:.1f}px | Steer: {steering:.2f} | Dist: {front_dist:.2f}m")
            else:
                rospy.loginfo_throttle(1, f"State: {self.state.name} | L/R: {L_near}/{R_near} | Error: {error_px:.1f}px | Steer: {steering:.2f} | Dist: {front_dist:.2f}m")

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