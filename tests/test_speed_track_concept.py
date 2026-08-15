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
import cv2
import numpy as np
import time
import math

try:
    import rospy
    from sensor_msgs.msg import LaserScan, Image
    HAS_ROS = True
except ImportError:
    rospy = None
    HAS_ROS = False

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
        
        # Biến dịch vạch ảo hiện tại (bằng pixel)
        self.current_offset_px = 0.0
        self.target_offset_px = 0.0

        # 4. Lưu dữ liệu cảm biến
        self.latest_scan = None
        self.latest_image = None

        # 5. Đăng ký ROS Topics
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        
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
            mid_x = int(w / 2)
            left_border = 0
            right_border = w - 1

            # Quét từ giữa ảnh sang bên TRÁI để tìm vạch trắng biên trái
            for x in range(mid_x, 0, -1):
                if thresh[y_line, x] == 255:
                    left_border = x
                    break
            
            # Quét từ giữa ảnh sang bên PHẢI để tìm vạch trắng biên phải
            for x in range(mid_x, w):
                if thresh[y_line, x] == 255:
                    right_border = x
                    break
            
            # Tâm đường là trung điểm của biên trái và biên phải
            center_x = int((left_border + right_border) / 2)
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
        """Đọc khoảng cách từ LiDAR ở cung trước mặt (-15 độ đến +15 độ)."""
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
            
            # Lọc trong khoảng phía trước xe (±15 độ)
            if -15.0 <= angle_deg <= 15.0:
                if msg.range_min < dist < msg.range_max:
                    distances.append(dist)
        
        return min(distances) if distances else float('inf')

    def is_side_clear(self):
        """Kiểm tra LiDAR bên sườn trái xe (70 đến 110 độ) để biết đã qua vật cản chưa."""
        if self.latest_scan is None:
            return True
        
        side_distances = []
        msg = self.latest_scan
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            angle_deg = angle_deg + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            # Quét vùng sườn bên trái xe (70 đến 110 độ)
            if 70.0 <= angle_deg <= 110.0:
                if msg.range_min < dist < msg.range_max:
                    side_distances.append(dist)
        
        if side_distances:
            # Nếu vật cản gần nhất bên hông trái xa hơn 40cm -> Coi như đã vượt qua an toàn
            return min(side_distances) > 0.40
        return True

    def update_fsm_and_offset(self, front_dist):
        """Quản lý các trạng thái né vật cản và điều chỉnh vạch ảo."""
        
        # --- STATE 1: ĐI THẲNG BÌNH THƯỜNG ---
        if self.state == 1:
            self.target_offset_px = 0.0  # Không dịch làn
            
            # Nếu phát hiện vật cản phía trước gần hơn cự ly kích hoạt
            if front_dist < self.TRIGGER_DIST:
                rospy.loginfo(f"⚠️ [FSM] PHÁT HIỆN VẬT CẢN: {front_dist:.2f}m. Chuyển sang TRẠNG THÁI NÉ!")
                self.state = 2  # Chuyển sang trạng thái 2 (Né)
                self.target_offset_px = self.DODGE_OFFSET_PX  # Yêu cầu dịch vạch ảo sang phải

        # --- STATE 2: ĐANG NÉ VẬT CẢN ---
        elif self.state == 2:
            self.target_offset_px = self.DODGE_OFFSET_PX
            
            # Kiểm tra sườn trái xe, nếu đã vượt qua hẳn vật cản
            if self.is_side_clear():
                rospy.loginfo("✅ [FSM] Đã vượt qua vật cản. Chuyển sang TRẠNG THÁI NHẬP LÀN CŨ!")
                self.state = 3  # Chuyển sang trạng thái 3 (Nhập làn)
                self.target_offset_px = 0.0  # Yêu cầu đưa vạch ảo về lại trung tâm

        # --- STATE 3: ĐANG NHẬP LẠI LÀN CŨ ---
        elif self.state == 3:
            self.target_offset_px = 0.0
            
            # Khi offset thực tế đã giảm hẳn về 0 (xe đã về giữa đường)
            if abs(self.current_offset_px) < 1.0:
                rospy.loginfo("🏠 [FSM] Đã về làn trung tâm an toàn. Quay lại TRẠNG THÁI BÁM LÀN.")
                self.state = 1  # Trở lại trạng thái bình thường

        # --- CƠ CHẾ RAMPING OFFSET (Dịch chuyển vạch ảo mượt mà) ---
        # Tránh việc bẻ góc lái đột ngột làm xe bị giật hoặc lật bánh
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
            
            # 4. Áp dụng vạch ảo đã dịch chuyển vào tâm đường gần
            # e_pixel: độ lệch giữa tâm xe (giữa ảnh) và tâm đường (đã cộng offset né)
            # Nếu current_offset_px dương -> Tâm xe dịch phải -> Xe đánh lái sang phải
            target_center_near = C_near + self.current_offset_px
            error = target_center_near - (img_w / 2.0)
            
            # 5. Bộ điều khiển P đơn giản để tính góc lái (Calibrate Kp trên xe thật)
            # Sai số càng lớn xe bẻ lái càng mạnh
            Kp = 0.006
            steering = error * Kp
            
            # Giới hạn góc lái tối đa trong khoảng [-1.0, 1.0]
            steering = max(-1.0, min(1.0, steering))
            
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
