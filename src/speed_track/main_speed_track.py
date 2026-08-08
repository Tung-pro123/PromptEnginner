#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson AI Racer Challenge 2026 - Speed Track (Bài 1)
Hệ thống điều khiển hoàn toàn tự động không sử dụng bản đồ (Mapless Autonomous).
Tái cấu trúc theo hạ tầng Phân lớp (Layered Architecture) & Blackboard Pattern.

- Xử lý ảnh Dual-Filter (HSV Red + White Background) cho xa hình mới.
- Gom cụm & Gán nhãn biên theo FSM State-Aware Segment Clustering.
- Máy trạng thái FSM 3 bước (NORMAL, DODGING, REENTERING) với Safety Steering Override (>= 0.28).
- Giao thức Trả làn 2 Giai đoạn (Two-Stage Re-entering: Open-loop 0.50 trong 1.2s).
- Tùy chọn chuyển đổi linh hoạt giữa P-Controller và Optimal LQR Controller.

Chạy trên xe:
    python3 src/speed_track/main_speed_track.py
"""

import sys
import os

# Sắp xếp lại sys.path để ưu tiên thư viện Python 3
sys.path = [p for p in sys.path if 'python2.7' not in p]

import rospy
import cv2
import numpy as np
import time
import csv
from sensor_msgs.msg import LaserScan, Image

# Import các mô-đun phân lớp tập trung
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import settings
from src.core.blackboard import Blackboard
from src.perception.camera.camera_processor import CameraProcessor
from src.perception.lidar.lidar_processor import LidarProcessor
from src.fsm.fsm_manager import FSMManager, RobotState
from src.control.pid_controller import PIDController
from src.control.lqr_controller import LQRController
from src.core.control.racer_controller import RacerController

class SpeedTrackController:
    def __init__(self):
        rospy.init_node('speed_track_controller_node', anonymous=True)
        rospy.loginfo("=== KHỞI TẠO BỘ ĐIỀU KHIỂN SPEED TRACK (LAYERED ARCHITECTURE) ===")
        
        self.blackboard = Blackboard()
        self.camera_processor = CameraProcessor(self.blackboard)
        self.lidar_processor = LidarProcessor(self.blackboard)
        self.fsm = FSMManager(self.blackboard)
        self.racer = RacerController()
        
        if settings.CONTROLLER_TYPE == 'lqr':
            self.controller = LQRController(self.blackboard)
            rospy.loginfo("-> Đã chọn Bộ điều khiển Tối ưu LQR.")
        else:
            self.controller = PIDController(self.blackboard)
            rospy.loginfo("-> Đã chọn Bộ điều khiển P-Controller.")

        self.latest_scan = None
        self.latest_image = None
        self.video_writer = None
        self.csv_file = None
        self.csv_writer = None

        self.setup_ros_subscribers()
        self.setup_logging()

    def setup_ros_subscribers(self):
        rospy.Subscriber(settings.ROS_TOPIC_CAMERA, Image, self.camera_callback, queue_size=1)
        rospy.Subscriber(settings.ROS_TOPIC_LIDAR, LaserScan, self.lidar_callback, queue_size=1)
        rospy.loginfo("[ROS] Đã đăng ký Subscriptions thành công.")

    def camera_callback(self, msg):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'bgr8':
                self.latest_image = img.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'rgb8':
                img_rgb = img.reshape((msg.height, msg.width, 3))
                self.latest_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                self.latest_image = cv2.cvtColor(img.reshape((msg.height, msg.width)), cv2.COLOR_GRAY2BGR)
        except Exception as e:
            rospy.logerr(f"Lỗi đọc camera ROS: {e}")

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def setup_logging(self):
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(settings.VIDEO_OUTPUT_FILENAME, fourcc, 20.0, (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
            
            self.csv_file = open(settings.CSV_DEBUG_FILENAME, mode='w')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'front_dist', 'closest_angle', 'state', 'dodge_dir', 'offset_px', 'C_near', 'steering', 'throttle'])
            rospy.loginfo("[LOGGING] Đã tạo file Video & CSV debug.")
        except Exception as e:
            rospy.logwarn(f"Không thể tạo logging: {e}")

    def run(self):
        rate = rospy.Rate(20) # 20 Hz
        rospy.loginfo("=== SẴN SÀNG CHẠY XE SPEED TRACK ===")
        
        start_loop_time = time.time()
        
        while not rospy.is_shutdown():
            if self.latest_image is None:
                rate.sleep()
                continue

            now = time.time()

            # 1. Perception - LiDAR Processing
            front_dist = self.lidar_processor.get_front_obstacle_distance(self.latest_scan)
            closest_angle, _ = self.lidar_processor.get_closest_obstacle_angle_in_range(self.latest_scan, settings.FRONT_ANGLE_MIN, settings.FRONT_ANGLE_MAX, settings.TRIGGER_DIST)

            # 2. FSM State Transition
            state, dodge_direction, current_offset_px = self.fsm.update(front_dist, closest_angle, self.lidar_processor, self.latest_scan)

            # 3. Perception - Vision Processing (Dual-Filter & State-Aware)
            reenter_duration = (now - self.fsm.reenter_start_time) if state == RobotState.STATE_REENTERING else 0.0
            C_near, C_far, y_near, y_far, debug_frame = self.camera_processor.process_frame(
                self.latest_image, fsm_state=state, dodge_direction=dodge_direction, current_offset_px=current_offset_px
            )

            # 4. Control - Calculate Steering & Throttle
            if settings.CONTROLLER_TYPE == 'lqr':
                steering = self.controller.compute_steering(C_near, C_far, y_near, y_far, settings.BASE_SPEED, current_offset_px)
            else:
                steering = self.controller.compute_steering(C_near, state, dodge_direction, current_offset_px, reenter_duration)

            throttle = settings.BASE_SPEED
            self.racer.set_steering(steering)
            self.racer.set_throttle(throttle)

            # 5. Logging & Visualizing
            if debug_frame is not None:
                debug_frame = self.lidar_processor.draw_lidar_radar(debug_frame, self.latest_scan)
                
                # Hiển thị HUD thông số
                state_text = f"STATE: {self.fsm.get_state_name()} | OFF: {current_offset_px:.1f}px"
                cv2.putText(debug_frame, state_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(debug_frame, f"STEER: {steering:.2f} | DIST: {front_dist:.2f}m", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                if self.video_writer:
                    self.video_writer.write(debug_frame)

            if self.csv_writer:
                self.csv_writer.writerow([now - start_loop_time, front_dist, closest_angle, self.fsm.get_state_name(), dodge_direction, current_offset_px, C_near, steering, throttle])

            rate.sleep()

    def shutdown(self):
        rospy.loginfo("=== DỪNG XE VÀ GIẢI PHÓNG TÀI NGUYÊN ===")
        self.racer.stop()
        if self.video_writer:
            self.video_writer.release()
        if self.csv_file:
            self.csv_file.close()

if __name__ == '__main__':
    controller = SpeedTrackController()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        controller.shutdown()