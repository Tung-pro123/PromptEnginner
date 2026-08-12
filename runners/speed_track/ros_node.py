#!/usr/bin/env python3
import sys
sys.path.append("../../")

import os

# Sắp xếp lại sys.path để ưu tiên thư viện Python 3, tránh xung đột với ROS Python 2.7
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rospy
from sensor_msgs.msg import LaserScan, Image
import cv2
import numpy as np
import math

from robot.config import settings
from robot.fsm.fsm_manager import FSMManager
from robot.control.pid_controller import PIDController
from robot.control.predictive_controller import PredictiveController
from robot.perception.camera_processor import CameraProcessor
from robot.perception.lidar_processor import LidarProcessor
from robot.debug.debugger import Debugger

from robot.utils.blackboard import Blackboard

class ROSSpeedTrackNode:
    """Node ROS sử dụng kiến trúc module để chạy robot (Blackboard Pattern)"""
    def __init__(self):
        rospy.init_node('speed_track_modular_node', anonymous=True)
        
        self.blackboard = Blackboard()
        
        # Khởi tạo các module cốt lõi (Knowledge Sources)
        self.fsm = FSMManager()
        
        controller_type = getattr(settings, 'CONTROLLER_TYPE', 'pid')
        if controller_type == 'predictive':
            self.controller = PredictiveController(self.blackboard)
            rospy.loginfo("Sử dụng PredictiveController.")
        else:
            self.controller = PIDController(self.blackboard)
            rospy.loginfo("Sử dụng PIDController.")
        
        self.camera = CameraProcessor(self.blackboard)
        self.lidar = LidarProcessor(self.blackboard)
        self.debugger = Debugger(debug_mode=True)
        
        self.controller.initialize()
        # self.camera.initialize()
        self.lidar.initialize()
        
        # ROS Subscribers
        rospy.Subscriber(settings.ROS_TOPIC_LIDAR, LaserScan, self.lidar.ros_callback)
        rospy.Subscriber(settings.ROS_TOPIC_CAMERA, Image, self.camera.ros_callback)
        
        rospy.loginfo("Node ROS Speed Track (Blackboard) đã khởi động thành công.")
        
    def run(self):
        """Vòng lặp điều khiển chính chạy ở 20Hz"""
        rate = rospy.Rate(20)
        
        try:
            while not rospy.is_shutdown():
                if not self.blackboard.has('latest_image') or not self.blackboard.has('latest_scan'):
                    rospy.logwarn_throttle(2, "Đang chờ dữ liệu từ Camera và Lidar...")
                    rate.sleep()
                    continue
                    
                # Các Processor xử lý theo thứ tự (Knowledge Sources)
                self.lidar.process(self.blackboard)
                self.fsm.process(self.blackboard)
                self.camera.process(self.blackboard)
                self.controller.process(self.blackboard)

                # Debugger: ghi CSV, video và in toàn bộ log debug (tập trung ở đây)
                self.debugger.process(self.blackboard)

                rate.sleep()

        except KeyboardInterrupt:
            rospy.logwarn("Đã nhận tín hiệu Ctrl+C từ người dùng (KeyboardInterrupt)!")

    def stop(self):
        """Đóng an toàn khi người dùng nhấn Ctrl+C"""
        rospy.loginfo("--- BẮT ĐẦU DỪNG HỆ THỐNG ---")
        if hasattr(self, 'controller') and self.controller:
            rospy.loginfo("Xả ga, trả lái về 0...")
            self.controller.stop()
        if hasattr(self, 'debugger') and self.debugger:
            rospy.loginfo("Tắt các cửa sổ debug...")
            self.debugger.close()
        rospy.loginfo("--- ĐÃ DỪNG AN TOÀN ---")

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from robot.utils.error_logger import log_crash

    node = None
    try:
        node = ROSSpeedTrackNode()
        node.run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
    except Exception as e:
        log_crash("ros_speed_track", e)
        raise
    finally:
        if node:
            node.stop()

