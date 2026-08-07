#!/usr/bin/env python3
import sys
sys.path.append("../")

import os
import rospy
from sensor_msgs.msg import Joy, Image

# Sắp xếp lại sys.path để ưu tiên thư viện Python 3, tránh xung đột với ROS Python 2.7
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

from src.config import settings
from src.control.pid_controller import PIDController
from src.core.blackboard import Blackboard
from src.perception.camera.camera_processor import CameraProcessor
from src.debug.debugger import Debugger

class ROSJoyTeleopNode:
    """Node ROS sử dụng tay cầm (Joystick/Gamepad) để điều khiển xe"""
    def __init__(self):
        rospy.init_node('joy_teleop_node', anonymous=True)
        
        # Mượn PIDController làm driver để gửi lệnh xuống phần cứng (NvidiaRacecar)
        self.controller = PIDController()
        self.controller.initialize()
        
        # Khởi tạo các công cụ Debug & Camera
        self.blackboard = Blackboard()
        self.camera = CameraProcessor(self.blackboard)
        self.debugger = Debugger(debug_mode=True)
        
        # Đăng ký nhận dữ liệu từ topic
        rospy.Subscriber(settings.ROS_TOPIC_JOY, Joy, self.joy_callback)
        rospy.Subscriber(settings.ROS_TOPIC_CAMERA, Image, self.camera.ros_callback)
        
        self.steering = 0.0
        self.throttle = 0.0
        self.turbo_mode = False
        
        rospy.loginfo("Node Teleop Joystick đã khởi động.")
        rospy.loginfo("HDSD: Cần trái (lên/xuống) để chỉnh GA, Cần phải (trái/phải) để bẻ LÁI.")
        rospy.loginfo("Nhấn giữ R1 để kích hoạt chế độ chạy nhanh (Turbo Mode).")
        
    def joy_callback(self, msg):
        """Xử lý tín hiệu tay cầm mỗi khi nhận được tin nhắn ROS"""
        try:
            # Trục (Axes) - Phụ thuộc vào chuẩn tay cầm (PS4/Xbox)
            # Thông thường:
            # msg.axes[1]: Cần trái lên/xuống (1.0 = Lên max, -1.0 = Xuống max)
            # msg.axes[3]: Cần phải trái/phải (1.0 = Trái max, -1.0 = Phải max)
            
            raw_throttle = msg.axes[1] if len(msg.axes) > 1 else 0.0
            raw_steering = msg.axes[3] if len(msg.axes) > 3 else 0.0
            
            # Nút (Buttons): msg.buttons[5] thường là R1 (Bumper Phải)
            self.turbo_mode = (msg.buttons[5] == 1) if len(msg.buttons) > 5 else False
            
            # Tính toán giá trị ga an toàn
            max_thr = settings.MAX_THROTTLE if self.turbo_mode else settings.BASE_SPEED
            self.throttle = raw_throttle * max_thr
            
            # Góc lái tay cầm thường ngược một chút so với trục tọa độ, có thể cần đảo dấu (-)
            self.steering = raw_steering * settings.MAX_STEERING
            
        except Exception as e:
            rospy.logerr(f"Lỗi đọc Joystick: {e}")

    def run(self):
        """Vòng lặp gửi lệnh liên tục xuống xe"""
        rate = rospy.Rate(20) # Cập nhật 20Hz
        try:
            while not rospy.is_shutdown():
                # Bỏ qua FSM, truyền thẳng lệnh điều khiển xuống motor
                self.controller.move(self.throttle, self.steering)
                
                # Ghi dữ liệu vào Blackboard để Debugger xuất ra video/csv
                self.blackboard.set('state_name', 'TELEOP')
                self.blackboard.set('steering', self.steering)
                self.blackboard.set('throttle', self.throttle)
                
                # Xử lý camera (nếu có frame mới)
                self.camera.process(self.blackboard)
                
                # Xuất log và video
                self.debugger.process(self.blackboard)
                
                rate.sleep()
                
        except KeyboardInterrupt:
            rospy.logwarn("Nhận tín hiệu dừng từ người dùng.")
            
    def stop(self):
        """Tắt động cơ an toàn"""
        if hasattr(self, 'controller') and self.controller:
            self.controller.stop()
        if hasattr(self, 'debugger') and self.debugger:
            self.debugger.close()
        rospy.loginfo("Đã dừng an toàn.")

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from error_logger import log_crash

    node = None
    try:
        node = ROSJoyTeleopNode()
        node.run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
    except Exception as e:
        log_crash("ros_joy_teleop", e)
        raise
    finally:
        if node:
            node.stop()

