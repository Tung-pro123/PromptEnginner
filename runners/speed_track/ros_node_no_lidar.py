#!/usr/bin/env python3
import sys
import os
import traceback

print("[STARTUP] Bắt đầu khởi động ros_speed_track_no_lidar.py...", flush=True)

sys.path.append("../../")

# Sắp xếp lại sys.path để ưu tiên thư viện Python 3, tránh xung đột với ROS Python 2.7
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

def _global_excepthook(exc_type, exc_value, exc_tb):
    """Bắt tất cả lỗi chưa được xử lý và in ra console."""
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"[UNCAUGHT ERROR] {exc_type.__name__}: {exc_value}", file=sys.stderr, flush=True)
    print(tb_str, file=sys.stderr, flush=True)
    print(f"{'='*60}", file=sys.stderr, flush=True)
    # Ghi vào file crash.log nếu có thể
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "crash_no_lidar.log"), "a") as f:
            import datetime
            f.write(f"\n[{datetime.datetime.now()}]\n{tb_str}\n")
    except Exception:
        pass
sys.excepthook = _global_excepthook

print("[STARTUP] Importing ROS và các thư viện...", flush=True)
try:
    import rospy
    from sensor_msgs.msg import Image
    import cv2
    import numpy as np
    import math
    print("[STARTUP] Import cơ bản OK.", flush=True)
except Exception as _e:
    print(f"[STARTUP ERROR] Không thể import thư viện cơ bản: {_e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)

print("[STARTUP] Importing các module src...", flush=True)
try:
    from robot.config import settings
    from robot.fsm.fsm_manager import CameraFSMManager
    from robot.control.pid_controller import PIDController
    from robot.control.predictive_controller import PredictiveController
    from robot.perception.camera_processor import CameraProcessor
    from robot.debug.debugger import Debugger
    from robot.utils.blackboard import Blackboard
    print("[STARTUP] Import src module OK.", flush=True)
except Exception as _e:
    print(f"[STARTUP ERROR] Lỗi import src module: {_e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)

class ROSSpeedTrackNoLidarNode:
    """Node ROS sử dụng kiến trúc module để chạy robot (Không dùng Lidar)"""
    def __init__(self):
        rospy.init_node('speed_track_no_lidar_node', anonymous=True)
        
        self.blackboard = Blackboard()
        self.debugger = Debugger(debug_mode=True)
        
        # Khởi tạo các module cốt lõi (Knowledge Sources)
        self.fsm = CameraFSMManager()
        
        controller_type = getattr(settings, 'CONTROLLER_TYPE', 'pid')
        if controller_type == 'predictive':
            self.controller = PredictiveController(self.blackboard)
            self.debugger._info("Sử dụng PredictiveController.")
        else:
            self.controller = PIDController(self.blackboard)
            self.debugger._info("Sử dụng PIDController.")
        
        self.camera = CameraProcessor(self.blackboard)
        
        self.controller.initialize()
        self.camera.initialize()
        
        # ROS Subscribers (Chỉ dùng Camera)
        rospy.Subscriber(settings.ROS_TOPIC_CAMERA, Image, self.camera.ros_callback)
        
        self.debugger._info("Node ROS Speed Track (No Lidar) đã khởi động thành công.")
        
    def run(self):
        """Vòng lặp điều khiển chính chạy ở 20Hz"""
        rate = rospy.Rate(20)
        
        try:
            while not rospy.is_shutdown():
                # Bỏ điều kiện chờ Lidar, chỉ chờ Camera
                if not self.blackboard.has('latest_image'):
                    self.debugger._info("Đang chờ dữ liệu từ Camera...")
                    rate.sleep()
                    continue
                    
                # Các Processor xử lý theo thứ tự
                # Camera phải chạy trước để FSM có dữ liệu waypoints mà phán đoán
                self.camera.process(self.blackboard)
                self.fsm.process(self.blackboard)
                self.controller.process(self.blackboard)

                # Debugger: ghi CSV, video và in toàn bộ log debug (tập trung ở đây)
                self.debugger.process(self.blackboard)

                rate.sleep()

        except KeyboardInterrupt:
            if hasattr(self, 'debugger') and self.debugger:
                self.debugger._info("Đã nhận tín hiệu Ctrl+C từ người dùng (KeyboardInterrupt)!")
        except Exception as e:
            if hasattr(self, 'debugger') and self.debugger:
                self.debugger.log_error(e, "Lỗi hệ thống trong vòng lặp chính")

    def stop(self):
        """Đóng an toàn khi người dùng nhấn Ctrl+C"""
        if hasattr(self, 'debugger') and self.debugger:
            self.debugger._info("--- BẮT ĐẦU DỪNG HỆ THỐNG ---")
        if hasattr(self, 'controller') and self.controller:
            if hasattr(self, 'debugger') and self.debugger:
                self.debugger._info("Xả ga, trả lái về 0...")
            self.controller.stop()
        if hasattr(self, 'debugger') and self.debugger:
            self.debugger._info("Tắt các cửa sổ debug...")
            self.debugger.close()
            self.debugger._info("--- ĐÃ DỪNG AN TOÀN ---")

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from robot.utils.error_logger import log_crash

    print("[STARTUP] Khởi tạo Node...", flush=True)
    node = None
    try:
        node = ROSSpeedTrackNoLidarNode()
        print("[STARTUP] Node đã khởi tạo xong. Bắt đầu chạy...", flush=True)
        node.run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        print("[SHUTDOWN] Nhận tín hiệu thoát (Ctrl+C / ROS shutdown).", flush=True)
    except Exception as e:
        log_crash("ros_speed_track_no_lidar", e)
        raise  # Ném lỗi lên để hệ thống thấy exit code != 0
    finally:
        if node:
            node.stop()
        print("[SHUTDOWN] Thoát chương trình.", flush=True)

