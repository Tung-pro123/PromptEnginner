#!/usr/bin/env python3
import sys
sys.path.append("../")

import os
import rospy
import random
import time

# Sắp xếp lại sys.path để ưu tiên thư viện Python 3, tránh xung đột với ROS Python 2.7
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

from sensor_msgs.msg import Image
from src.config import settings
from src.control.pid_controller import PIDController
from src.core.blackboard import Blackboard
from src.perception.camera.camera_processor import CameraProcessor
from src.debug.debugger import Debugger

class ROSRandomWalkNode:
    """Node ROS điều khiển robot chạy ngẫu nhiên"""
    def __init__(self):
        rospy.init_node('random_walk_node', anonymous=True)
        
        # Khởi tạo các công cụ Debug, Camera & Blackboard
        self.blackboard = Blackboard()
        self.camera = CameraProcessor(self.blackboard)
        self.debugger = Debugger(debug_mode=True)
        
        self.controller = PIDController(self.blackboard)
        self.controller.initialize()
        
        # Đăng ký nhận dữ liệu từ topic Camera
        rospy.Subscriber(settings.ROS_TOPIC_CAMERA, Image, self.camera.ros_callback)
        
        self.steering = 0.0
        self.throttle = 0.0
        
        rospy.loginfo("Node ROS Random Walk đã khởi động. Robot sẽ chạy ngẫu nhiên!")
        
    def run(self):
        """Vòng lặp điều khiển chính"""
        rate = rospy.Rate(10) # Cập nhật 10Hz
        last_change_time = time.time()
        
        try:
            while not rospy.is_shutdown():
                current_time = time.time()
                
                # Thay đổi hành động ngẫu nhiên mỗi 2 giây
                if current_time - last_change_time > 2.0:
                    # Tạo ngẫu nhiên throttle từ -0.3 đến 0.3 (để xe đi chậm, an toàn)
                    self.throttle = random.uniform(-0.3, 0.3)
                    
                    # Tạo ngẫu nhiên steering từ -1.0 đến 1.0 (trái đến phải)
                    self.steering = random.uniform(-1.0, 1.0)
                    
                    last_change_time = current_time
                    rospy.loginfo(f"Hành động mới - Throttle: {self.throttle:.2f}, Steering: {self.steering:.2f}")
                
                # Truyền lệnh điều khiển xuống motor
                self.controller.move(self.throttle, self.steering)
                
                # Ghi dữ liệu vào Blackboard để Debugger xuất ra video/csv
                self.blackboard.set('state_name', 'RANDOM_WALK')
                self.blackboard.set('steering', self.steering)
                self.blackboard.set('throttle', self.throttle)
                
                # Xử lý camera (nếu có frame mới)
                self.camera.process(self.blackboard)
                
                # Xuất log và video
                self.debugger.process(self.blackboard)
                
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.logwarn("Đã nhận tín hiệu Ctrl+C từ người dùng!")

    def stop(self):
        """Tắt động cơ an toàn"""
        rospy.loginfo("--- BẮT ĐẦU DỪNG HỆ THỐNG ---")
        if hasattr(self, 'controller') and self.controller:
            self.controller.stop()
        if hasattr(self, 'debugger') and self.debugger:
            self.debugger.close()
        rospy.loginfo("--- ĐÃ DỪNG AN TOÀN ---")

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from error_logger import log_crash

    node = None
    try:
        node = ROSRandomWalkNode()
        node.run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
    except Exception as e:
        log_crash("ros_random_walk", e)
        raise
    finally:
        if node:
            node.stop()

