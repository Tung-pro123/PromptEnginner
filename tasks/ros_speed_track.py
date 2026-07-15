#!/usr/bin/env python3
import sys
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

from src.config import settings
from src.fsm.fsm_manager import FSMManager
from src.control.pid_controller import PIDController
from src.perception.camera.camera_processor import CameraProcessor
from src.perception.lidar.lidar_processor import LidarProcessor
from src.debug.debugger import Debugger

from src.core.blackboard import Blackboard

class ROSSpeedTrackNode:
    """Node ROS sử dụng kiến trúc module để chạy robot (Blackboard Pattern)"""
    def __init__(self):
        rospy.init_node('speed_track_modular_node', anonymous=True)
        
        self.blackboard = Blackboard()
        
        # Khởi tạo các module cốt lõi (Knowledge Sources)
        self.fsm = FSMManager()
        self.controller = PIDController(self.blackboard)
        self.camera = CameraProcessor(self.blackboard)
        self.lidar = LidarProcessor(self.blackboard)
        self.debugger = Debugger(debug_mode=True)
        
        self.controller.initialize()
        self.camera.initialize()
        self.lidar.initialize()
        
        # ROS Subscribers
        rospy.Subscriber('/scan', LaserScan, self.lidar.ros_callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera.ros_callback)
        
        rospy.loginfo("Node ROS Speed Track (Blackboard) đã khởi động thành công.")
        
    def run(self):
        """Vòng lặp điều khiển chính chạy ở 20Hz"""
        rate = rospy.Rate(20)
        
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
            self.debugger.process(self.blackboard)
            
            # Lấy data để in log ROS (tùy chọn)
            state_name = self.blackboard.get('state_name', 'UNKNOWN')
            offset = self.blackboard.get('current_offset_px', 0.0)
            steer = self.blackboard.get('steering', 0.0)
            f_dist = self.blackboard.get('front_dist', 999.0)
            
            rospy.loginfo_throttle(1, f"[{state_name}] offset: {offset:.1f}, steer: {steer:.3f}, f_dist: {f_dist:.2f}")
            
            rate.sleep()

        # Đóng an toàn khi người dùng nhấn Ctrl+C
        rospy.loginfo("Dừng hệ thống, xả ga và tắt lưu log.")
        self.controller.stop()
        self.debugger.close()

if __name__ == '__main__':
    try:
        node = ROSSpeedTrackNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"Lỗi ngoài ý muốn: {e}")
        try:
            r = PIDController()
            r.stop()
        except:
            pass
