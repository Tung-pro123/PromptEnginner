#!/usr/bin/env python3
import sys
sys.path.append("../../")

import os
import rospy
from sensor_msgs.msg import Joy, Image

# Sắp xếp lại sys.path để ưu tiên thư viện Python 3, tránh xung đột với ROS Python 2.7
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

from robot.config import settings
from robot.control.pid_controller import PIDController
from robot.utils.blackboard import Blackboard
from robot.perception.camera_processor import CameraProcessor
from robot.debug.debugger import Debugger

class ROSJoyTeleopNode:
    """Node ROS sử dụng tay cầm (Joystick/Gamepad) để điều khiển xe"""
    def __init__(self):
        rospy.init_node('joy_teleop_node', anonymous=True)
        
        # Mượn PIDController làm driver để gửi lệnh xuống phần cứng (NvidiaRacecar)
        self.controller = PIDController()
        self.controller.initialize()
        
        # Cấu hình phần cứng: Tăng hệ số bẻ lái (steering_gain) để góc cua lớn hơn
        if hasattr(self.controller, 'car') and self.controller.car:
            if hasattr(self.controller.car, 'steering_gain'):
                self.controller.car.steering_gain = 0.8
                rospy.loginfo(f"Đã cấu hình steering_gain = {self.controller.car.steering_gain}")
        
        # Khởi tạo các công cụ Debug & Camera (Đặt debug_mode=True để ghi video)
        self.blackboard = Blackboard()
        self.camera = CameraProcessor(self.blackboard)
        self.debugger = Debugger(debug_mode=True)
        self.last_print_time = 0.0
        
        # Đăng ký nhận dữ liệu từ topic
        rospy.Subscriber(settings.ROS_TOPIC_JOY, Joy, self.joy_callback)
        rospy.Subscriber(settings.ROS_TOPIC_CAMERA, Image, self.camera.ros_callback)
        
        self.steering = 0.0
        self.target_steering = 0.0
        self.steering_accel_step = 0.15  # Tốc độ bẻ lái mượt (càng nhỏ càng bẻ chậm, 0.15 tốn khoảng ~0.3s để quay kịch vô lăng)
        self.throttle = 0.0
        self.raw_throttle_input = 0.0
        self.raw_steering_input = 0.0
        self.turbo_mode = False
        self.e_stop_active = False
        self.speed_trim = 0.0
        self.dpad_up_pressed = False
        self.dpad_down_pressed = False
        
        rospy.loginfo("Node Teleop Joystick đã khởi động.")
        rospy.loginfo("HDSD: Lái (Cần gạt trái - Left Stick), Ga/Phanh (Cò R2/L2).")
        rospy.loginfo("Ga và Lái đều nhận diện lực bấm/độ nghiêng (Analog) để điều khiển tốc độ và góc rẽ tương ứng.")
        rospy.loginfo("Nhấn R1 để Turbo, Nhấn B (Circle) để E-Stop, Y (Triangle) để mở khóa E-Stop.")
        rospy.loginfo("D-pad lên/xuống để chỉnh tăng/giảm tốc độ giới hạn cơ bản.")
        
    def joy_callback(self, msg):
        """Xử lý tín hiệu tay cầm mỗi khi nhận được tin nhắn ROS"""
        try:
            # E-Stop logic
            if len(msg.buttons) > 1 and msg.buttons[1] == 1:
                self.e_stop_active = True
                rospy.logwarn(">>> ĐÃ KÍCH HOẠT E-STOP (Khóa động cơ) <<<")
            if len(msg.buttons) > 3 and msg.buttons[3] == 1:
                self.e_stop_active = False
                rospy.loginfo(">>> ĐÃ MỞ KHÓA E-STOP <<<")
                
            # Còi (L3)
            if len(msg.buttons) > 9 and msg.buttons[9] == 1:
                rospy.loginfo("BÍP BÍP! (Còi)")
            
            # Speed Trim logic (D-Pad Up/Down)
            if len(msg.axes) > 7:
                dpad_y = msg.axes[7]
                if dpad_y == 1.0 and not self.dpad_up_pressed:
                    self.speed_trim += 0.05
                    self.dpad_up_pressed = True
                    rospy.loginfo(f"Tăng tốc độ cơ bản lên: {settings.BASE_SPEED + self.speed_trim:.2f}")
                elif dpad_y == -1.0 and not self.dpad_down_pressed:
                    self.speed_trim -= 0.05
                    self.dpad_down_pressed = True
                    rospy.loginfo(f"Giảm tốc độ cơ bản xuống: {settings.BASE_SPEED + self.speed_trim:.2f}")
                elif dpad_y == 0.0:
                    self.dpad_up_pressed = False
                    self.dpad_down_pressed = False

            # Ga / Lùi (Cò L2/R2) - Dựa vào lực bấm (từ 0.0 đến 1.0)
            l2 = (1.0 - msg.axes[2]) / 2.0 if len(msg.axes) > 2 else 0.0
            r2 = (1.0 - msg.axes[5]) / 2.0 if len(msg.axes) > 5 else 0.0
            
            self.raw_throttle_input = r2 - l2

            # Lái (Cần gạt trái - Left Stick X) - Dựa vào độ nghiêng cần gạt
            # Trái/Phải của Left Stick thường là axes[0]
            self.raw_steering_input = msg.axes[0] if len(msg.axes) > 0 else 0.0
            
            self.turbo_mode = (msg.buttons[5] == 1) if len(msg.buttons) > 5 else False
            
            if self.e_stop_active:
                self.throttle = 0.0
                self.raw_throttle_input = 0.0
                self.target_steering = 0.0

            current_time = rospy.get_time()
            if current_time - self.last_print_time >= 0.5:
                status_str = "KHOÁ E-STOP" if self.e_stop_active else "CHẠY"
                rospy.loginfo(f"[JOYSTICK {status_str}] Ga = {self.throttle:.2f} | Lái = {self.steering:.2f} | Turbo = {self.turbo_mode}")
                self.last_print_time = current_time
            
        except Exception as e:
            rospy.logerr(f"Lỗi đọc Joystick: {e}")

    def run(self):
        """Vòng lặp gửi lệnh liên tục xuống xe"""
        rate = rospy.Rate(20) # Cập nhật 20Hz
        try:
            while not rospy.is_shutdown():
                # Bỏ qua FSM, truyền thẳng lệnh điều khiển xuống motor
                if self.e_stop_active:
                    self.controller.stop()
                    self.throttle = 0.0
                else:
                    # Phương pháp tích lũy tốc độ (Cruise Control Style)
                    # Tính giới hạn tốc độ hiện tại
                    base_spd = max(0.0, settings.BASE_SPEED + self.speed_trim)
                    max_thr = settings.MAX_THROTTLE if self.turbo_mode else base_spd
                    
                    # Cộng dồn vận tốc. Hệ số 0.02 tức là cần bóp đầy cò R2 trong khoảng 1s để đạt tốc độ 0.4
                    accel_step = 0.02 * self.raw_throttle_input
                    self.throttle += accel_step
                    
                    # Cắt tốc độ không vượt quá giới hạn
                    self.throttle = max(-max_thr, min(max_thr, self.throttle))
                    
                    # Tính góc lái mục tiêu dựa trên độ nghiêng cần gạt
                    self.target_steering = self.raw_steering_input * settings.MAX_STEERING
                    
                    # Bẻ lái từ từ (Smooth Steering) thay vì giật cục
                    if self.steering < self.target_steering:
                        self.steering = min(self.target_steering, self.steering + self.steering_accel_step)
                    elif self.steering > self.target_steering:
                        self.steering = max(self.target_steering, self.steering - self.steering_accel_step)
                    
                    self.controller.move(self.throttle, self.steering)
                
                # Ghi dữ liệu vào Blackboard để Debugger xuất ra video/csv
                self.blackboard.set('state_name', 'E-STOP' if self.e_stop_active else 'TELEOP')
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
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from robot.utils.error_logger import log_crash

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

