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
import csv
from enum import Enum
from sensor_msgs.msg import LaserScan, Image

# Import các module điều khiển nội bộ
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.control.pid_controller import PIDController
from src.core.control.lqr_controller import LQRController, ObstacleDetector

class RobotState(Enum):
    STATE_NORMAL = 1       # Bám làn bình thường
    STATE_DODGING = 2      # Đang lách né vật cản sang phải
    STATE_REENTERING = 3   # Đang lượn quay trở lại làn cũ

class SpeedTrackController:
    def __init__(self):
        rospy.init_node('speed_track_controller_node', anonymous=True)
        rospy.loginfo("=== KHỞI TẠO BỘ ĐIỀU KHIỂN SPEED TRACK (KHÔNG DÙNG MAP) ===")
        self.setup_parameters()
        self.initialize_hardware()

        # Trạng thái ban đầu
        self.state = RobotState.STATE_NORMAL
        self.latest_scan = None
        self.latest_image = None
        self.state_change_time = rospy.get_time()
        self.clear_counter = 0  # Bộ đếm lọc nhiễu tránh trả lái sớm
        self.dodge_direction = 1.0  # 1.0 = Lách phải (vật cản bên trái), -1.0 = Lách trái (vật cản bên phải)

        # Khởi tạo CSV Debug log
        self.CSV_FILENAME = 'speed_track_debug.csv'
        try:
            self.csv_file = open(self.CSV_FILENAME, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            # Viết tiêu đề cột cho file CSV
            self.csv_writer.writerow([
                'timestamp', 'state', 'front_dist', 
                'closest_angle', 'closest_dist', 
                'clear_counter', 'dodge_direction', 'current_offset_px', 'steering'
            ])
            rospy.loginfo(f"Khởi tạo file log CSV thành công: {self.CSV_FILENAME}")
        except Exception as e:
            rospy.logerr(f"Không thể mở file CSV: {e}")
            self.csv_file = None
            self.csv_writer = None

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
        self.DODGE_OFFSET_PX = 70      # Độ rộng dịch vạch ảo để né tránh (pixel) - đã tăng lên để né rộng hơn
        self.RAMP_STEP_PX = 5          # Tốc độ dịch chuyển vạch ảo mỗi frame (pixel/frame)
        self.SIDE_CLEAR_DIST = 0.45    # Khoảng cách sườn trái an toàn trước khi nhập làn (m)
        self.MIN_DODGE_TIME = 2.0      # Thời gian tối thiểu (giây) giữ góc né lách qua vật cản

        # 3. Kích thước ảnh xử lý
        self.WIDTH = 300
        self.HEIGHT = 300

        # 4. Trạng thái dịch làn ảo hiện tại và tự động hiệu chỉnh chiều rộng làn đường
        self.current_offset_px = 0.0
        self.target_offset_px = 0.0
        self.estimated_lane_width = 240.0  # Chiều rộng làn đường ước lượng mặc định (pixel)
        self.last_known_direction = 0.0    # Hướng lệch làn gần nhất: 1.0 (lệch phải/lane ở phải), -1.0 (lệch trái/lane ở trái)

        # 5. Cấu hình video ghi lại để phân tích lỗi sau lượt chạy
        self.VIDEO_OUTPUT_FILENAME = 'speed_track_run.avi'
        self.VIDEO_FPS = 20
        self.VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')

    def initialize_hardware(self):
        """Khởi tạo điều khiển phần cứng qua PIDController."""
        self.racer = PIDController()
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
            left_border = 0
            right_border = self.WIDTH - 1
            found_left = False
            found_right = False

            # Quét từ mép TRÁI ảnh (x = 0) đi vào trong để tìm biên trái
            for x in range(0, int(self.WIDTH / 2)):
                if thresh[y_line, x] == 255:
                    left_border = x
                    found_left = True
                    break
            
            # Quét từ mép PHẢI ảnh (x = w - 1) đi vào trong để tìm biên phải
            for x in range(self.WIDTH - 1, int(self.WIDTH / 2), -1):
                if thresh[y_line, x] == 255:
                    right_border = x
                    found_right = True
                    break
            
            # Khôi phục biên đơn nếu mất một trong hai bên:
            if found_left and found_right:
                # Tìm thấy cả 2 biên -> Tính toán và hiệu chuẩn chiều rộng làn đường thực tế
                width = right_border - left_border
                if 160 < width < 280:
                    self.estimated_lane_width = 0.9 * self.estimated_lane_width + 0.1 * width
                center_x = int((left_border + right_border) / 2)
                # Lưu lại hướng lệch gần nhất
                self.last_known_direction = np.sign(center_x - self.WIDTH / 2.0)
            elif found_left:
                # Chỉ thấy biên trái -> Dựng biên phải ảo dựa trên chiều rộng làn đường ước lượng
                right_border = int(left_border + self.estimated_lane_width)
                center_x = int(left_border + self.estimated_lane_width / 2)
                self.last_known_direction = np.sign(center_x - self.WIDTH / 2.0)
            elif found_right:
                # Chỉ thấy biên phải -> Dựng biên trái ảo dựa trên chiều rộng làn đường ước lượng
                left_border = int(right_border - self.estimated_lane_width)
                center_x = int(right_border - self.estimated_lane_width / 2)
                self.last_known_direction = np.sign(center_x - self.WIDTH / 2.0)
            else:
                # Mất cả 2 biên -> Sử dụng hướng lệch đã biết gần nhất để đánh lái quay trở lại làn
                center_x = int(self.WIDTH / 2.0 + self.last_known_direction * (self.estimated_lane_width / 4.0))
                
            # Giới hạn giá trị biên nằm trong ảnh
            left_border = max(0, min(self.WIDTH - 1, left_border))
            right_border = max(0, min(self.WIDTH - 1, right_border))
            center_x = max(0, min(self.WIDTH - 1, center_x))
            
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
        """Đo khoảng cách vật cản trước mặt bằng LiDAR (nới rộng ra -35 đến +35 độ để tránh lọt vật cản lệch)."""
        if self.latest_scan is None:
            return float('inf')
        
        distances = []
        msg = self.latest_scan
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            angle_deg = angle_deg + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            # Quét rộng ra ±35 độ để không bỏ sót các vật cản nằm hơi chéo
            if -35.0 <= angle_deg <= 35.0:
                if msg.range_min < dist < msg.range_max:
                    distances.append(dist)
                    
        return min(distances) if distances else float('inf')

    def get_closest_obstacle_angle_in_range(self, min_angle_deg, max_angle_deg, max_dist=0.80):
        """Tìm góc và khoảng cách của vật cản gần nhất trong một dải quét cụ thể."""
        if self.latest_scan is None:
            return None, float('inf')
        
        min_dist = float('inf')
        closest_angle = None
        msg = self.latest_scan
        
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            angle_deg = angle_deg + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            if min_angle_deg <= angle_deg <= max_angle_deg:
                if msg.range_min < dist < max_dist:
                    if dist < min_dist:
                        min_dist = dist
                        closest_angle = angle_deg
                        
        return closest_angle, min_dist

    def update_fsm_states(self, front_dist):
        """Cập nhật máy trạng thái FSM né tránh vật cản động theo 2 phía (trái/phải)."""
        
        # --- STATE 1: BÁM LÀN BÌNH THƯỜNG ---
        if self.state == RobotState.STATE_NORMAL:
            self.target_offset_px = 0.0
            self.clear_counter = 0  # Reset bộ lọc nhiễu
            
            # Quét tìm vật cản bất kỳ trong hình nêm trước mặt để xác định hướng lách tránh
            closest_angle, closest_dist = self.get_closest_obstacle_angle_in_range(-35.0, 35.0, max_dist=0.80)
            self.last_closest_angle = closest_angle
            self.last_closest_dist = closest_dist
            
            # Kích hoạt né tránh khi khoảng cách trước mặt bé hơn ngưỡng an toàn
            if front_dist < self.TRIGGER_DIST:
                # Quyết định hướng né tránh động dựa trên góc của vật cản:
                # Nếu góc < 0 (vật cản nằm bên phải): Lách sang trái (dodge_direction = -1.0)
                # Nếu góc >= 0 (vật cản nằm bên trái hoặc chính giữa): Lách sang phải (dodge_direction = 1.0)
                if closest_angle is not None and closest_angle < 0.0:
                    self.dodge_direction = -1.0
                    direction_str = "TRÁI"
                else:
                    self.dodge_direction = 1.0
                    direction_str = "PHẢI"
                
                rospy.loginfo(f"⚠️ [FSM] PHÁT HIỆN VẬT CẢN: {front_dist:.2f}m ở góc {closest_angle:.1f} độ. LÁCH SANG {direction_str}!")
                self.state = RobotState.STATE_DODGING
                self.state_change_time = rospy.get_time()
                self.target_offset_px = self.dodge_direction * self.DODGE_OFFSET_PX

        # --- STATE 2: ĐANG LÁCH NÉ VẬT CẢN ---
        elif self.state == RobotState.STATE_DODGING:
            self.target_offset_px = self.dodge_direction * self.DODGE_OFFSET_PX
            
            # Lọc góc dựa theo hướng đang né:
            # - Nếu né sang PHẢI (vật cản bên trái): Theo dõi hông bên trái xe [-30, 150]
            # - Nếu né sang TRÁI (vật cản bên phải): Theo dõi hông bên phải xe [-150, 30]
            if self.dodge_direction == 1.0:
                closest_angle, closest_dist = self.get_closest_obstacle_angle_in_range(-30.0, 150.0, max_dist=0.80)
                is_clear = (closest_angle is None or closest_angle > 110.0)
            else:
                closest_angle, closest_dist = self.get_closest_obstacle_angle_in_range(-150.0, 30.0, max_dist=0.80)
                is_clear = (closest_angle is None or closest_angle < -110.0)
                
            self.last_closest_angle = closest_angle
            self.last_closest_dist = closest_dist
            
            # Đếm lọc nhiễu liên tục trong 8 frames
            if is_clear:
                self.clear_counter += 1
            else:
                self.clear_counter = 0
            
            # Watchdog giám sát thời gian né tối đa 3.5s để tránh đâm biên
            time_in_state = rospy.get_time() - self.state_change_time
            is_timeout = time_in_state > 3.5
            
            if (time_in_state > 0.5 and self.clear_counter >= 8) or is_timeout:
                if is_timeout:
                    rospy.logwarn(f"🕒 [FSM] Hết thời gian né tối đa (3.5s). Kích hoạt Watchdog quay về làn chính.")
                else:
                    rospy.loginfo(f"✅ [FSM] Đã vượt qua vật cản an toàn (Lọc đạt {self.clear_counter}). Quay về làn chính.")
                self.state = RobotState.STATE_REENTERING
                self.state_change_time = rospy.get_time()
                self.target_offset_px = 0.0

        # --- STATE 3: NHẬP LẠI LÀN CŨ ---
        elif self.state == RobotState.STATE_REENTERING:
            self.target_offset_px = 0.0
            closest_angle, closest_dist = self.get_closest_obstacle_angle_in_range(-150.0, 150.0, max_dist=0.80)
            self.last_closest_angle = closest_angle
            self.last_closest_dist = closest_dist
            
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

            # 6. Truyền lệnh điều khiển ga/lái trực tiếp xuống xe qua PIDController (I2C)
            self.racer.steer(steering, self.BASE_SPEED)

            # 7. Ghi video debug để phân tích lượt chạy
            if self.video_writer is not None:
                # Vẽ thêm thông tin FSM và khoảng cách lên ảnh để tiện phân tích video
                cv2.putText(debug_img, f"State: {self.state.name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(debug_img, f"Dist: {front_dist:.2f}m", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(debug_img, f"Offset: {self.current_offset_px:.1f}px", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                self.video_writer.write(debug_img)

            # 8. Ghi dữ liệu hành trình vào file CSV Debug
            if self.csv_writer is not None:
                try:
                    self.csv_writer.writerow([
                        rospy.get_time(),
                        self.state.value,
                        front_dist,
                        self.last_closest_angle if hasattr(self, 'last_closest_angle') else None,
                        self.last_closest_dist if hasattr(self, 'last_closest_dist') else float('inf'),
                        self.clear_counter,
                        self.dodge_direction,
                        self.current_offset_px,
                        steering
                    ])
                except Exception as e:
                    rospy.logerr(f"Lỗi ghi CSV: {e}")

            # In log debug ra màn hình Terminal
            rospy.loginfo_throttle(1, f"State: {self.state.name} | Dist: {front_dist:.2f}m | Offset: {self.current_offset_px:.1f}px | Steer: {steering:.2f}")

            rate.sleep()

        # Dừng xe khi thoát chương trình
        self.racer.stop()
        if self.video_writer is not None:
            self.video_writer.release()
            rospy.loginfo("Đã lưu video debug.")
        if self.csv_file is not None:
            self.csv_file.close()
            rospy.loginfo(f"Đã lưu và đóng file log CSV: {self.CSV_FILENAME}")
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
            r = PIDController()
            r.stop()
        except:
            pass
        if 'controller' in locals() and controller.csv_file is not None:
            controller.csv_file.close()