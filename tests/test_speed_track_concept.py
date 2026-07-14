#!/usr/bin/env python3
"""
DỰ THẢO CODE THỬ NGHIỆM SPEED TRACK: Bám vạch biên & Né vật cản (FSM)

Script này giải quyết 2 bài toán lớn của bạn:
1. Bám làn an toàn: Dựa vào việc tìm khoảng trống lòng đường đen nằm giữa 2 vạch trắng biên
   (giúp xe giữ khoảng cách đều với 2 biên, tuyệt đối không chạm biên để tránh bị loại).
2. Quy trình chuyển trạng thái né vật cản: Máy trạng thái (FSM) 3 bước kết hợp LiDAR.

Chạy trên xe:
    python3 tests/test_speed_track_concept.py
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
from sensor_msgs.msg import LaserScan, Image

# Thêm thư mục cha vào path để import RacerController
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.control.racer_controller import RacerController

class SpeedTrackConcept:
    def __init__(self):
        rospy.init_node('speed_track_concept_node', anonymous=True)
        rospy.loginfo("=== Khởi tạo Node Thử Nghiệm Speed Track ===")

        # 1. Khởi tạo cơ cấu lái và ga
        self.racer = RacerController()
        self.racer.stop()

        # 2. Các tham số điều khiển (Cần calibrate trên xe thật)
        self.BASE_SPEED = 0.18        # Tốc độ ga chạy thử (giới hạn an toàn)
        self.TRIGGER_DIST = 0.65      # Khoảng cách kích hoạt né vật cản (65cm)
        self.DODGE_OFFSET_PX = 60     # Số pixel dịch chuyển làn ảo sang phải để né (tương đương ~22cm thực tế)
        self.RAMP_STEP_PX = 5         # Tốc độ dịch chuyển vạch ảo mỗi frame (tránh bẻ lái gấp làm lật xe)

        # 3. Trạng thái xe (FSM)
        # STATE_NORMAL: 1 - Bám làn thẳng bình thường
        # STATE_DODGING: 2 - Phát hiện vật cản, đang đánh lái dịch phải để lách qua
        # STATE_REENTERING: 3 - Đã vượt qua vật cản, đang lái mượt mà quay về làn cũ
        self.state = 1
        self.clear_counter = 0
        
        # Biến dịch vạch ảo hiện tại (bằng pixel) và chiều rộng làn đường ước lượng
        self.current_offset_px = 0.0
        self.target_offset_px = 0.0
        self.estimated_lane_width = 240.0
        self.last_known_direction = 0.0

        # 4. Lưu dữ liệu cảm biến
        self.latest_scan = None
        self.latest_image = None

        # 5. Đăng ký ROS Topics
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        
        rospy.loginfo("Hệ thống đã sẵn sàng. Chờ nhận dữ liệu cảm biến...")

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def camera_callback(self, msg):
        """Chuyển đổi ảnh ROS sang OpenCV."""
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'bgr8':
                self.latest_image = img.reshape((msg.height, msg.width, 3))
        except Exception as e:
            rospy.logerr(f"Lỗi nhận ảnh camera: {e}")

    # =========================================================================
    # GIẢI QUYẾT VẤN ĐỀ 1: THUẬT TOÁN BÁM LÀN (LINE FOLLOWING) AN TOÀN
    # =========================================================================
    def get_lane_centers(self, frame):
        """
        Thuật toán bám làn đường bằng cách quét tìm 2 vạch biên trắng:
        - Quét từ tâm ảnh ra 2 bên để tìm điểm chuyển tiếp Đen -> Trắng đầu tiên.
        - Trung điểm của 2 điểm đó chính là tâm lòng đường đen an toàn nhất.
        - Tránh việc xe đè lên vạch biên gây loại trực tiếp.
        """
        h, w = frame.shape[:2]
        
        # Chọn 2 dòng quét: Dòng quét GẦN xe (để căn lệch làn) và Dòng quét XA xe (để lấy hướng cua)
        y_near = int(h * 0.85)  # Dòng quét ở 85% chiều cao ảnh (gần xe)
        y_far = int(h * 0.55)   # Dòng quét ở 55% chiều cao ảnh (xa xe)

        # Chuyển sang ảnh xám và nhị phân hóa để làm nổi bật vạch trắng biên
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        def find_borders_at_y(y_line):
            # Tìm tất cả điểm trắng trên dòng quét
            white_xs = [x for x in range(w) if thresh[y_line, x] == 255]
            
            segments = []
            if len(white_xs) > 0:
                current_segment = [white_xs[0]]
                for x in white_xs[1:]:
                    if x - current_segment[-1] > 15:
                        segments.append(int(np.mean(current_segment)))
                        current_segment = [x]
                    else:
                        current_segment.append(x)
                segments.append(int(np.mean(current_segment)))

            left_border = 0
            right_border = w - 1
            found_left = False
            found_right = False

            if len(segments) >= 2:
                left_border = segments[0]
                right_border = segments[-1]
                found_left = True
                found_right = True
            elif len(segments) == 1:
                x_val = segments[0]
                # Sử dụng phân loại phân vùng (Zone-based) kết hợp với FSM Prior để tránh đổi góc lái đột ngột:
                if x_val < 110:
                    left_border = x_val
                    found_left = True
                elif x_val > 190:
                    right_border = x_val
                    found_right = True
                else:
                    # Vùng trung tâm (110 <= x_val <= 190): Sử dụng trạng thái né tránh làm định hướng
                    if self.state in [2, 3]:  # 2: STATE_DODGING, 3: STATE_REENTERING
                        if self.dodge_direction == -1.0:
                            left_border = x_val
                            found_left = True
                        else:
                            right_border = x_val
                            found_right = True
                    else:
                        if x_val < w / 2.0:
                            left_border = x_val
                            found_left = True
                        else:
                            right_border = x_val
                            found_right = True

            # Khôi phục biên đơn nếu mất một bên:
            if found_left and found_right:
                width = right_border - left_border
                if 160 < width < 280:
                    self.estimated_lane_width = 0.9 * self.estimated_lane_width + 0.1 * width
                center_x = int((left_border + right_border) / 2)
                self.last_known_direction = np.sign(center_x - w / 2.0)
            elif found_left:
                right_border = int(left_border + self.estimated_lane_width)
                center_x = int(left_border + self.estimated_lane_width / 2)
                self.last_known_direction = np.sign(center_x - w / 2.0)
            elif found_right:
                left_border = int(right_border - self.estimated_lane_width)
                center_x = int(right_border - self.estimated_lane_width / 2)
                self.last_known_direction = np.sign(center_x - w / 2.0)
            else:
                # Mất cả 2 biên -> Sử dụng hướng lệch đã biết gần nhất để đánh lái quay trở lại làn
                center_x = int(w / 2.0 + self.last_known_direction * (self.estimated_lane_width / 4.0))
                
            left_border = max(0, min(w - 1, left_border))
            right_border = max(0, min(w - 1, right_border))
            center_x = max(0, min(w - 1, center_x))
            
            return center_x, left_border, right_border

        # Tính toán cho cả 2 vùng quét
        C_near, L_near, R_near = find_borders_at_y(y_near)
        C_far, L_far, R_far = find_borders_at_y(y_far)

        # Vẽ hình minh họa debug (lưu vào file video sau này)
        debug_frame = frame.copy()
        cv2.line(debug_frame, (0, y_near), (w, y_near), (0, 255, 255), 1)
        cv2.line(debug_frame, (0, y_far), (w, y_far), (0, 255, 255), 1)
        # Vẽ biên trái/phải màu đỏ
        cv2.circle(debug_frame, (L_near, y_near), 5, (0, 0, 255), -1)
        cv2.circle(debug_frame, (R_near, y_near), 5, (0, 0, 255), -1)
        # Vẽ tâm đường màu xanh lá
        cv2.circle(debug_frame, (C_near, y_near), 6, (0, 255, 0), -1)
        cv2.circle(debug_frame, (C_far, y_far), 6, (0, 255, 0), -1)

        return C_near, C_far, y_near, y_far, debug_frame

    # =========================================================================
    # GIẢI QUYẾT VẤN ĐỀ 2: QUY TRÌNH CHUYỂN TRẠNG THÁI NÉ VẬT CẢN (FSM)
    # =========================================================================
    def get_front_distance(self):
        """Đọc khoảng cách từ LiDAR ở cung trước mặt (±35 độ)."""
        if self.latest_scan is None:
            return float('inf')
        
        distances = []
        msg = self.latest_scan
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            # Bù góc 180 độ do LiDAR lắp ngược trên xe
            angle_deg = angle_deg + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            # Lọc trong khoảng phía trước xe (±35 độ)
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

    def update_fsm_and_offset(self, front_dist):
        """Quản lý các trạng thái né vật cản và điều chỉnh vạch ảo."""
        
        # --- STATE 1: ĐI THẲNG BÌNH THƯỜNG ---
        if self.state == 1:
            self.target_offset_px = 0.0  # Không dịch làn
            self.clear_counter = 0
            
            # Quét tìm vật cản trước mặt
            closest_angle, closest_dist = self.get_closest_obstacle_angle_in_range(-35.0, 35.0, max_dist=0.80)
            
            # Nếu phát hiện vật cản phía trước gần hơn cự ly kích hoạt (đã đảo ngược theo mirrored LiDAR)
            if front_dist < self.TRIGGER_DIST:
                # - Góc >= 0 (vật cản bên phải trong mirrored): Lách sang TRÁI (dodge_direction = -1.0)
                # - Góc < 0 (vật cản bên trái trong mirrored): Lách sang PHẢI (dodge_direction = 1.0)
                if closest_angle is not None and closest_angle >= 0.0:
                    self.dodge_direction = -1.0
                    direction_str = "TRÁI"
                else:
                    self.dodge_direction = 1.0
                    direction_str = "PHẢI"
                
                rospy.loginfo(f"⚠️ [FSM] PHÁT HIỆN VẬT CẢN: {front_dist:.2f}m ở góc {closest_angle:.1f} độ. LÁCH SANG {direction_str}!")
                self.state = 2  # Chuyển sang trạng thái 2 (Né)
                self.state_change_time = rospy.get_time()
                self.target_offset_px = self.dodge_direction * self.DODGE_OFFSET_PX  # Yêu cầu dịch vạch ảo
 
        # --- STATE 2: ĐANG NÉ VẬT CẢN ---
        elif self.state == 2:
            self.target_offset_px = self.dodge_direction * self.DODGE_OFFSET_PX
            
            # Lọc góc dựa theo hướng đang né (đã đảo ngược theo mirrored LiDAR):
            # - Nếu né sang PHẢI: Theo dõi hông bên trái (góc âm trong mirrored: -150.0 đến 30.0)
            # - Nếu né sang TRÁI: Theo dõi hông bên phải (góc dương trong mirrored: -30.0 đến 150.0)
            if self.dodge_direction == 1.0:
                closest_angle, closest_dist = self.get_closest_obstacle_angle_in_range(-150.0, 30.0, max_dist=0.80)
                is_clear = (closest_angle is None or closest_angle < -110.0)
            else:
                closest_angle, closest_dist = self.get_closest_obstacle_angle_in_range(-30.0, 150.0, max_dist=0.80)
                is_clear = (closest_angle is None or closest_angle > 110.0)
            
            # Kiểm tra bộ đếm lọc nhiễu trước khi xác nhận đã vượt qua hẳn
            if is_clear:
                self.clear_counter += 1
            else:
                self.clear_counter = 0
                
            time_in_state = rospy.get_time() - self.state_change_time
            is_timeout = time_in_state > 3.5
            
            if (time_in_state > 0.5 and self.clear_counter >= 8) or is_timeout:
                rospy.loginfo(f"✅ [FSM] Đã vượt qua vật cản. Chuyển sang TRẠNG THÁI NHẬP LÀN CŨ!")
                self.state = 3  # Chuyển sang trạng thái 3 (Nhập làn)
                self.state_change_time = rospy.get_time()
                self.target_offset_px = 0.0  # Yêu cầu đưa vạch ảo về lại trung tâm

        # --- STATE 3: ĐANG NHẬP LẠI LÀN CŨ ---
        elif self.state == 3:
            self.target_offset_px = 0.0
            
            # Đảm bảo thời gian trả lái kéo dài ít nhất 2.5 giây để xe ổn định
            time_in_reentering = rospy.get_time() - self.state_change_time
            if abs(self.current_offset_px) < 1.0 and time_in_reentering >= 2.5:
                rospy.loginfo("🏠 [FSM] Đã về làn trung tâm an toàn. Quay lại TRẠNG THÁI BÁM LÀN.")
                self.state = 1  # Trở lại trạng thái bình thường

        # --- CƠ CHẾ RAMPING OFFSET (Dịch chuyển vạch ảo mượt mà) ---
        # Tránh việc bẻ góc lái đột ngột làm xe bị giật hoặc lật bánh
        diff = self.target_offset_px - self.current_offset_px
        if abs(diff) > 0.1:
            # Trả làn chậm (2.0 px/frame) để lướt mượt, né nhanh (5.0 px/frame) để tránh vật cản gấp
            ramp_step = 2.0 if self.state == 3 else self.RAMP_STEP_PX
            step = np.sign(diff) * ramp_step
            if abs(step) > abs(diff):
                self.current_offset_px = self.target_offset_px
            else:
                self.current_offset_px += step
        else:
            self.current_offset_px = self.target_offset_px

    # =========================================================================
    # VÒNG LẶP ĐIỀU KHIỂN CHÍNH
    # =========================================================================
    def run(self):
        rate = rospy.Rate(20) # Chạy vòng lặp 20 FPS
        
        while not rospy.is_shutdown():
            if self.latest_image is None:
                rospy.logwarn_throttle(5, "Đang chờ ảnh từ Camera...")
                rate.sleep()
                continue
            
            # 1. Tìm thông tin đường (Tâm đường và các biên)
            C_near, C_far, y_near, y_far, debug_img = self.get_lane_centers(self.latest_image)
            img_w = self.latest_image.shape[1]
            
            # 2. Đo khoảng cách vật cản trước mặt
            front_dist = self.get_front_distance()
            
            # 3. Cập nhật trạng thái FSM và tính toán vạch ảo dịch chuyển
            self.update_fsm_and_offset(front_dist)
            
            # Kiểm tra xem có đang trong giai đoạn đầu ép lái trả làn mở (open-loop pre-steer) hay không
            is_open_loop_return = False
            if self.state == 3:  # STATE_REENTERING
                time_in_reentering = rospy.get_time() - self.state_change_time
                if time_in_reentering < 1.2:
                    is_open_loop_return = True

            if is_open_loop_return:
                # Ép lái góc cố định mạnh hơn (0.50) ngược hướng né tránh để chủ động xoay đầu xe về giữa đường
                steering = -0.50 * self.dodge_direction
            else:
                # 4. Áp dụng vạch ảo đã dịch chuyển vào tâm đường gần
                # e_pixel: độ lệch giữa tâm xe (giữa ảnh) và tâm đường (đã cộng offset né)
                target_center_near = C_near + self.current_offset_px
                error = target_center_near - (img_w / 2.0)
                
                # 5. Bộ điều khiển P đơn giản để tính góc lái
                Kp = 0.006
                steering = error * Kp
                
                # Giới hạn góc lái tối đa trong khoảng [-1.0, 1.0]
                steering = max(-1.0, min(1.0, steering))
                
                # Thiết lập góc lái tối thiểu bắt buộc trong trạng thái né tránh (Safety Override)
                if self.state == 2:  # STATE_DODGING
                    min_dodge_steer = 0.28
                    if self.dodge_direction == 1.0:
                        steering = max(min_dodge_steer, steering)
                    elif self.dodge_direction == -1.0:
                        steering = min(-min_dodge_steer, steering)
            
            # 6. Truyền lệnh xuống động cơ xe
            self.racer.steer(steering, self.BASE_SPEED)
            
            # Log thông tin debug thời gian thực
            print(f"Trạng thái FSM: {self.state} | Vật cản trước: {front_dist:.2f}m | Dịch làn ảo: {self.current_offset_px:.1f}px | Lái: {steering:.2f}")
            
            # Lưu ảnh debug hoặc hiển thị nếu cần (Jupyter)
            # cv2.imwrite("/home/jetson/Desktop/Admin/captured_images/debug_lane.jpg", debug_img)
            
            rate.sleep()
            
        self.racer.stop()
        print("Đã dừng xe.")

if __name__ == '__main__':
    try:
        concept = SpeedTrackConcept()
        concept.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
        # Dừng xe khẩn cấp khi gặp lỗi code
        try:
            r = RacerController()
            r.stop()
        except:
            pass
