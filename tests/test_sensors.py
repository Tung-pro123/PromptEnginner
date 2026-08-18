#!/usr/bin/env python3
"""
Chương trình chẩn đoán và kiểm tra các cảm biến khác trên xe JetRacer Pro:
1. Camera (Kiểm tra đọc trực tiếp GStreamer và đọc qua ROS topic).
2. LiDAR (Kiểm tra tần số nhận tin nhắn và đo khoảng cách xung quanh).
3. IMU (Kiểm tra xem có nhận được dữ liệu gia tốc/góc nghiêng).

Cách chạy trên Jetson:
    python3 tests/test_sensors.py
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
from sensor_msgs.msg import LaserScan, Image, Imu

# Thêm thư mục gốc chứa src vào path để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SensorTester:
    def __init__(self):
        rospy.init_node('sensor_tester_node', anonymous=True)
        print("="*60)
        print("🔍 ĐANG KHỞI TẠO BỘ KIỂM TRA CẢM BIẾN JETRACER 🔍")
        print("="*60)

        # Biến đếm và đo lường
        self.camera_fps_count = 0
        self.camera_last_time = time.time()
        self.camera_hz = 0.0
        self.camera_resolution = "Chưa nhận"

        self.lidar_hz_count = 0
        self.lidar_last_time = time.time()
        self.lidar_hz = 0.0
        self.lidar_points_count = 0
        self.lidar_min_dist = float('inf')

        self.imu_hz_count = 0
        self.imu_last_time = time.time()
        self.imu_hz = 0.0
        self.imu_orientation = "Chưa nhận"

        # Đăng ký các Subscriber ROS
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        # Thử 2 topic camera phổ biến của JetRacer
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        # Topic IMU phổ biến
        rospy.Subscriber('/imu', Imu, self.imu_callback)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)

    def camera_callback(self, msg):
        self.camera_fps_count += 1
        t = time.time()
        dt = t - self.camera_last_time
        if dt >= 2.0:
            self.camera_hz = self.camera_fps_count / dt
            self.camera_fps_count = 0
            self.camera_last_time = t
            self.camera_resolution = f"{msg.width}x{msg.height} ({msg.encoding})"

    def lidar_callback(self, msg):
        self.lidar_hz_count += 1
        t = time.time()
        dt = t - self.lidar_last_time
        if dt >= 2.0:
            self.lidar_hz = self.lidar_hz_count / dt
            self.lidar_hz_count = 0
            self.lidar_last_time = t
        
        self.lidar_points_count = len(msg.ranges)
        # Lọc các khoảng cách hợp lệ
        valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid_ranges:
            self.lidar_min_dist = min(valid_ranges)

    def imu_callback(self, msg):
        self.imu_hz_count += 1
        t = time.time()
        dt = t - self.imu_last_time
        if dt >= 2.0:
            self.imu_hz = self.imu_hz_count / dt
            self.imu_hz_count = 0
            self.imu_last_time = t
        
        # Lấy thông số góc Quaternion
        q = msg.orientation
        self.imu_orientation = f"x:{q.x:.2f}, y:{q.y:.2f}, z:{q.z:.2f}, w:{q.w:.2f}"

    def test_direct_camera(self):
        """Thử mở trực tiếp CSI Camera bằng GStreamer của OpenCV (không qua ROS)."""
        print("\n--- Thử kết nối trực tiếp Camera bằng OpenCV GStreamer ---")
        
        # GStreamer pipeline cho Jetson Nano CSI Camera
        pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=(int)300, height=(int)300, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        )
        
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("  [OK] Camera CSI mở trực tiếp THÀNH CÔNG bằng OpenCV!")
            ret, frame = cap.read()
            if ret:
                print(f"  [OK] Đã chụp được khung hình trực tiếp. Kích thước: {frame.shape}")
                # Lưu thử 1 ảnh để kiểm tra chất lượng ống kính
                cv2.imwrite("test_camera_direct.jpg", frame)
                print("  -> Đã lưu ảnh test thành công vào file 'test_camera_direct.jpg'")
            cap.release()
        else:
            print("  [CẢNH BÁO] Không thể mở trực tiếp Camera qua GStreamer.")
            print("  -> Có thể camera đang bị một tiến trình khác (như ROS Camera node) chiếm dụng.")

    def run(self):
        # 1. Chạy thử camera trực tiếp
        self.test_direct_camera()
        
        print("\n--- Bắt đầu đọc dữ liệu cảm biến từ ROS (Chờ 5 giây)... ---")
        print("Lưu ý: Hãy chắc chắn bạn đã chạy 'roslaunch jetracer jetracer.launch' ở terminal khác.")
        
        # Chờ nhận tin nhắn
        start_time = time.time()
        while not rospy.is_shutdown() and time.time() - start_time < 5.0:
            rospy.sleep(0.1)
            
        print("\n" + "="*50)
        print("📊 BÁO CÁO SỨC KHỎE CẢM BIẾN JETRACER 📊")
        print("="*50)
        
        # 1. Báo cáo Camera
        if self.camera_hz > 0:
            print(f"📷 1. CAMERA (ROS): Đang hoạt động")
            print(f"  - Tần số nhận: {self.camera_hz:.1f} Hz (FPS)")
            print(f"  - Độ phân giải: {self.camera_resolution}")
        else:
            print("📷 1. CAMERA (ROS): [LỖI] Không nhận được tin nhắn trên topic camera.")
            print("  -> Hướng dẫn: Đảm bảo node camera của ROS đã chạy (ví dụ jetson_camera).")
            
        # 2. Báo cáo LiDAR
        if self.lidar_hz > 0:
            print(f"\n📡 2. LiDAR (RPLIDAR): Đang hoạt động")
            print(f"  - Tần số quét: {self.lidar_hz:.1f} Hz")
            print(f"  - Số điểm quét/vòng: {self.lidar_points_count} điểm")
            print(f"  - Vật cản gần nhất quét được: {self.lidar_min_dist:.2f} mét")
        else:
            print("\n📡 2. LiDAR (RPLIDAR): [LỖI] Không nhận được tin nhắn trên topic /scan.")
            print("  -> Hướng dẫn: Đảm bảo LiDAR đã cắm cáp USB và node rplidar_node đã chạy.")
            
        # 3. Báo cáo IMU
        if self.imu_hz > 0:
            print(f"\n⚖️ 3. IMU (Góc nghiêng): Đang hoạt động")
            print(f"  - Tần số: {self.imu_hz:.1f} Hz")
            print(f"  - Góc Quaternion hiện tại: {self.imu_orientation}")
        else:
            print("\n⚖️ 3. IMU (Góc nghiêng): [LƯU Ý] Không nhận được tin nhắn IMU.")
            print("  -> Có thể xe của bạn không lắp cảm biến IMU rời, hoặc topic IMU khác tên.")
            
        print("="*50)

if __name__ == '__main__':
    try:
        tester = SensorTester()
        tester.run()
    except rospy.ROSInterruptException:
        pass
