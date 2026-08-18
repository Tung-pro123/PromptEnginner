#!/usr/bin/env python3
import rospy
import cv2
import csv
import time
import os
import numpy as np
from sensor_msgs.msg import Image, LaserScan, Joy
from cv_bridge import CvBridge

# Giả sử bạn import các module perception của bạn tại đây
# from robot.perception.segmentation import SegLaneDetector
# from robot.perception.traffic_detector import TrafficSignDetector
# from robot.perception.lidar_processor import LidarProcessor

class ImitationDataCollector:
    def __init__(self):
        rospy.init_node('imitation_data_collector', anonymous=True)
        
        self.bridge = CvBridge()
        
        # --- State Variables ---
        self.latest_image = None
        self.latest_scan = None
        
        # --- Action Variables (Continuous) ---
        self.current_steer = 0.0
        self.current_throttle = 0.0
        
        # --- Recording State ---
        self.is_recording = False
        self.record_button_idx = 0 # Ví dụ: Nút X trên PS4/Xbox
        
        # --- Logger Setup ---
        self.log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_file = None
        self.csv_writer = None
        
        # Khởi tạo Perception Modules (Placeholder - tuỳ chỉnh theo kiến trúc của bạn)
        # self.lane_detector = SegLaneDetector(model_path="...")
        # self.sign_detector = TrafficSignDetector(model_path="...")
        # self.lidar_processor = LidarProcessor()
        
        # --- Subscribers ---
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/joy', Joy, self.joy_callback)
        
        rospy.loginfo("Đã khởi tạo Node Thu Thập Dữ Liệu Học Bắt Chước (Imitation Learning).")
        rospy.loginfo("Nhấn nút 0 (A/X) trên tay cầm để Bắt đầu / Kết thúc ghi dữ liệu.")

    def start_recording(self):
        ts = time.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.log_dir, f'dataset_{ts}.csv')
        self.csv_file = open(filepath, 'w', newline='')
        
        # Định nghĩa các trường dữ liệu (State + Action)
        fieldnames = [
            'timestamp', 
            'lane_offset', 'lane_curvature', 
            'sign_class', 
            'lidar_dist_front', 'lidar_dist_side', 
            'action_steer', 'action_throttle'
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.is_recording = True
        rospy.loginfo(f"Đang GHI dữ liệu vào: {filepath}")

    def stop_recording(self):
        if self.csv_file:
            self.csv_file.close()
        self.is_recording = False
        rospy.loginfo("Đã DỪNG ghi dữ liệu.")

    def camera_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Lỗi đọc ảnh: {e}")

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def joy_callback(self, msg):
        # Giả sử trục 0 là steer (trái/phải), trục 1 (hoặc 4) là throttle (tiến/lùi)
        # Điều chỉnh index trục phù hợp với tay cầm thực tế của bạn
        self.current_steer = msg.axes[0] 
        self.current_throttle = msg.axes[1]
        
        # Nút chuyển đổi trạng thái ghi
        record_btn_pressed = msg.buttons[self.record_button_idx]
        
        # Toggle recording state
        if record_btn_pressed and not hasattr(self, '_btn_pressed_last_frame'):
            self._btn_pressed_last_frame = True
            if self.is_recording:
                self.stop_recording()
            else:
                self.start_recording()
        elif not record_btn_pressed:
            if hasattr(self, '_btn_pressed_last_frame'):
                del self._btn_pressed_last_frame

    def run(self):
        rate = rospy.Rate(30) # 30 Hz
        while not rospy.is_shutdown():
            if self.is_recording and self.latest_image is not None and self.latest_scan is not None:
                # 1. Trích xuất State từ Camera & LiDAR
                # lane_state = self.lane_detector.process(self.latest_image)
                # sign_state = self.sign_detector.process(self.latest_image)
                # dist_front, dist_side = self.lidar_processor.process(self.latest_scan)
                
                # Placeholder State (thay thế bằng kết quả thật từ perception)
                lane_offset = 0.0
                lane_curvature = 0.0
                sign_class = 0 # 0: none, 1: stop, 2: left, ...
                lidar_dist_front = 10.0
                lidar_dist_side = 5.0
                
                # 2. Ghi vào file CSV
                row = {
                    'timestamp': time.time(),
                    'lane_offset': f'{lane_offset:.4f}',
                    'lane_curvature': f'{lane_curvature:.4f}',
                    'sign_class': sign_class,
                    'lidar_dist_front': f'{lidar_dist_front:.2f}',
                    'lidar_dist_side': f'{lidar_dist_side:.2f}',
                    'action_steer': f'{self.current_steer:.4f}',
                    'action_throttle': f'{self.current_throttle:.4f}'
                }
                self.csv_writer.writerow(row)
                
                # 3. (Tuỳ chọn) Đẩy steer và throttle xuống /cmd_vel để xe chạy thật
                # self.robot_controller.steer(self.current_steer, self.current_throttle)

            # Debug camera
            if self.latest_image is not None:
                cv2.imshow("Debug Camera", self.latest_image)
                cv2.waitKey(1)

            rate.sleep()
            
        # Cleanup
        cv2.destroyAllWindows()
        if self.is_recording:
            self.stop_recording()

if __name__ == '__main__':
    try:
        collector = ImitationDataCollector()
        collector.run()
    except rospy.ROSInterruptException:
        pass
