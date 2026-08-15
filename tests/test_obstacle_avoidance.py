#!/usr/bin/env python3
"""
Script chạy thử nghiệm Trường hợp 2: Né vật cản trên đường thẳng
Sử dụng bộ điều khiển LQR, cảm biến LiDAR để phát hiện khoảng cách 
và Camera để bám làn đường thẳng.

Yêu cầu:
    - Đặt 1 hộp carton trên vạch trắng thẳng làm vật cản.
    - Chạy trên xe bằng cách SSH và kích hoạt ROS trước.

Chạy lệnh:
    python3 tests/test_obstacle_avoidance.py
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

# Thêm thư mục gốc chứa src vào path để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import các controller của chúng ta
from src.core.control.racer_controller import RacerController
from src.core.control.lqr_controller import LQRController, ObstacleDetector

class SpeedTrackTestNode:
    def __init__(self):
        rospy.init_node('speed_track_test_node', anonymous=True)
        rospy.loginfo("=== Khởi tạo Node Test Né Vật Cản (LQR) ===")
        
        # 1. Cấu hình phần cứng
        self.racer = RacerController()
        # Khởi tạo LQR: Chiều dài cơ sở xe ~ 0.18m
        self.lqr = LQRController(wheelbase=0.18, scale_factor=0.0015)
        # Khởi tạo cảm biến tránh vật cản: Thời gian phản ứng 0.5s, Khoảng cách an toàn 20cm
        self.detector = ObstacleDetector(reaction_time=0.5, safe_distance=0.20)
        
        # 2. Tham số chạy xe
        self.BASE_SPEED = 0.20       # Tốc độ ga mặc định (0.0 -> 1.0)
        self.DODGE_OFFSET = 0.22     # Độ lệch vạch ảo né vật cản (22 cm sang phải)
        
        # 3. Trạng thái xe
        # 0: Chờ vạch, 1: Bám làn thẳng, 2: Đang né vật cản, 3: Đang nhập lại làn
        self.state = 1 
        self.latest_scan = None
        self.latest_image = None
        
        # 4. Đăng ký ROS Topics
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        # CSI Camera Topic trên Jetson
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        
        rospy.loginfo("Đã kết nối với LiDAR và Camera. Sẵn sàng chạy!")

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def camera_callback(self, msg):
        """
        Nhận ảnh từ ROS và chuyển đổi thủ công sang OpenCV (không cần cv_bridge)
        để tương thích hoàn toàn với Python 3.
        """
        try:
            # Chuyển đổi dữ liệu byte thành mảng numpy
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

    def get_line_centers(self, frame):
        """
        Xử lý ảnh để tìm tâm vạch trắng (hoặc tâm vùng lòng đường đen).
        Trả về: C_near (tâm gần), C_far (tâm xa), Y_near, Y_far
        """
        h, w = frame.shape[:2]
        
        # Định nghĩa vùng quét ROI gần và xa
        y_near = int(h * 0.85)
        h_near = int(h * 0.10)
        y_far = int(h * 0.55)
        h_far = int(h * 0.10)
        
        # Chuyển ảnh sang Grayscale (hoặc HSV) để tìm vạch trắng
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Threshold lọc vạch trắng (vạch trắng có độ sáng cao > 200)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # 1. Tìm tâm vùng gần (C_near)
        roi_near = thresh[y_near:y_near+h_near, :]
        M_near = cv2.moments(roi_near)
        c_near = int(M_near["m10"] / M_near["m00"]) if M_near["m00"] > 0 else int(w / 2)
        
        # 2. Tìm tâm vùng xa (C_far)
        roi_far = thresh[y_far:y_far+h_far, :]
        M_far = cv2.moments(roi_far)
        c_far = int(M_far["m10"] / M_far["m00"]) if M_far["m00"] > 0 else int(w / 2)
        
        return c_near, c_far, y_near + int(h_near/2), y_far + int(h_far/2)

    def check_side_clear(self, scan_msg):
        """Quét LiDAR bên sườn trái để xác nhận đã vượt qua vật cản chưa."""
        if scan_msg is None:
            return True
        # Quét góc sườn trái từ 70 độ đến 110 độ
        side_distances = []
        for i, dist in enumerate(scan_msg.ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            angle_deg = math.degrees(angle)
            # Bù 180 độ do góc xoay lắp đặt LiDAR ngược trên xe JetRacer
            angle_deg = angle_deg + 180.0
            angle_deg = (angle_deg + 180) % 360 - 180
            
            if 70.0 <= angle_deg <= 110.0:
                if scan_msg.range_min < dist < scan_msg.range_max:
                    side_distances.append(dist)
        
        if side_distances:
            # Nếu vật cản gần nhất bên hông xa hơn 40cm -> Coi như đã vượt qua
            return min(side_distances) > 0.40
        return True

    def run(self):
        rate = rospy.Rate(20) # Chạy vòng lặp ở tần số 20Hz
        
        rospy.loginfo("Bắt đầu chạy xe. Nhấn Ctrl+C để dừng khẩn cấp.")
        
        while not rospy.is_shutdown():
            if self.latest_image is None:
                rospy.logwarn_throttle(5, "Đang chờ dữ liệu camera...")
                rate.sleep()
                continue
                
            # 1. Trích xuất tâm vạch từ ảnh camera
            C_near, C_far, Y_near, Y_far = self.get_line_centers(self.latest_image)
            
            # 2. Tính khoảng cách tới vật cản phía trước từ LiDAR
            front_dist = self.detector.get_front_obstacle_distance(self.latest_scan)
            trigger_dist = self.detector.get_trigger_distance(self.BASE_SPEED)
            
            # 3. Điều khiển theo Máy trạng thái (FSM) né vật cản
            
            # --- TRẠNG THÁI 1: BÁM LÀN BÌNH THƯỜNG ---
            if self.state == 1:
                self.lqr.target_offset = 0.0 # Không né
                
                # Nếu phát hiện vật cản trước mặt gần hơn cự ly kích hoạt
                if front_dist < trigger_dist:
                    rospy.loginfo(f"⚠️ PHÁT HIỆN VẬT CẢN ở khoảng cách: {front_dist:.2f}m. Kích hoạt né!")
                    self.state = 2 # Chuyển sang trạng thái né
                    
            # --- TRẠNG THÁI 2: ĐANG NÉ VẬT CẢN (DỊCH PHẢI) ---
            elif self.state == 2:
                # Dịch vạch ảo sang phải để xe đánh lái né sang phải
                self.lqr.target_offset = self.DODGE_OFFSET
                
                # Kiểm tra xem hông bên trái đã trống hoàn toàn chưa (vượt qua vật cản)
                if self.check_side_clear(self.latest_scan):
                    rospy.loginfo("✅ Đã vượt qua vật cản. Chuẩn bị nhập lại làn cũ.")
                    self.state = 3 # Chuyển sang trạng thái nhập làn
                    
            # --- TRẠNG THÁI 3: NHẬP LẠI LÀN CŨ ---
            elif self.state == 3:
                # Đưa vạch ảo trở về vạch thật
                self.lqr.target_offset = 0.0
                
                # Khi offset thực tế đã giảm hẳn về 0
                if abs(self.lqr.current_offset) < 0.01:
                    rospy.loginfo("Đã nhập làn thành công. Quay lại chế độ bám vạch bình thường.")
                    self.state = 1 # Quay lại trạng thái 1
            
            # 4. Tính toán góc lái LQR
            # image_width của ảnh camera ngầm định là chiều rộng frame nhận được
            img_w = self.latest_image.shape[1]
            steering = self.lqr.compute_steering(C_near, C_far, Y_near, Y_far, self.BASE_SPEED, image_width=img_w)
            
            # 5. Truyền lệnh xuống xe
            self.racer.steer(steering, self.BASE_SPEED)
            
            # Hiển thị thông số debug
            rospy.loginfo_throttle(1, f"State: {self.state} | Dist: {front_dist:.2f}m | Offset: {self.lqr.current_offset:.2f}m | Steer: {steering:.2f}")
            
            rate.sleep()
            
        # Khi dừng chương trình, đảm bảo dừng xe an toàn
        self.racer.stop()
        rospy.loginfo("Đã dừng xe an toàn.")

if __name__ == '__main__':
    try:
        node = SpeedTrackTestNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
        # Dừng xe khẩn cấp
        try:
            r = RacerController()
            r.stop()
        except:
            pass
