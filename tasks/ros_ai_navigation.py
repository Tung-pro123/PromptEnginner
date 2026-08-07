#!/usr/bin/env python3
"""
ROS AI Navigation Node - Task chạy Robot với bộ não AI tích hợp
================================================================
Task này mở rộng từ ros_speed_track.py với tích hợp AIDecisionEngine,
cho phép robot tự quyết định:
  - Rẽ trái / Rẽ phải (tại ngã tư)
  - Đứng chờ (WAIT) khi bị chặn
  - Dừng khẩn cấp (EMERGENCY_STOP)
  - Né tránh (DODGE_LEFT / DODGE_RIGHT)
  - Lùi xe (REVERSE) khi bị kẹt
  - Bám làn (FOLLOW_LANE) trong điều kiện bình thường

Cách chạy trên Jetson Nano:
    python3 tasks/ros_ai_navigation.py
    python3 tasks/ros_ai_navigation.py --turn-priority right left straight

Kiến trúc xử lý (Pipeline mỗi chu kỳ 20Hz):
    LidarProcessor -> FSMManager -> CameraProcessor
        -> Controller (PID/Predictive) -> [AI Decision Engine]
        -> Thực thi lệnh ra Motor/Servo
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sắp xếp lại sys.path để ưu tiên thư viện Python 3
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

import rospy
from sensor_msgs.msg import LaserScan, Image
import cv2
import numpy as np

from src.config import settings
from src.fsm.fsm_manager import FSMManager
from src.control.pid_controller import PIDController
from src.control.predictive_controller import PredictiveController
from src.perception.camera.camera_processor import CameraProcessor
from src.perception.lidar.lidar_processor import LidarProcessor
from src.debug.debugger import Debugger
from src.core.blackboard import Blackboard
from src.ai.ai_decision_engine import AIDecisionEngine, Action
from src.perception.camera.traffic_detector import TrafficDetector


class ROSAINavigationNode:
    """
    Node ROS tích hợp AI Decision Engine.

    Kiến trúc: Blackboard Pattern + AI Brain Layer
    - Các processor (Camera, Lidar, Controller) vẫn chạy như cũ
    - AI Engine nhận output của Controller và quyết định lệnh cuối cùng
    - Nếu AI thấy cần rẽ/dừng/né -> override lệnh điều khiển
    - Nếu AI cho phép đi thẳng -> dùng nguyên lệnh từ Controller
    """

    def __init__(self, turn_priority=None):
        rospy.init_node('ai_navigation_node', anonymous=True)
        rospy.loginfo("[Init] ====== Khởi động AI Navigation Node ======")

        # --- Blackboard trung tâm ---
        rospy.loginfo("[Init] 1. Khởi tạo Blackboard...")
        self.blackboard = Blackboard()

        # --- Các module nhận thức (Perception) ---
        rospy.loginfo("[Init] 2. Khởi tạo FSM, Camera, Lidar...")
        self.fsm    = FSMManager()
        self.camera = CameraProcessor(self.blackboard)
        self.lidar  = LidarProcessor(self.blackboard)

        # --- Module debug ---
        rospy.loginfo("[Init] 3. Khởi tạo Debugger...")
        self.debugger = Debugger(debug_mode=True)

        # --- Controller cấp thấp (PID hoặc Predictive) ---
        controller_type = getattr(settings, 'CONTROLLER_TYPE', 'pid')
        rospy.loginfo(f"[Init] 4. Controller loại: {controller_type}")
        if controller_type == 'predictive':
            self.controller = PredictiveController(self.blackboard)
        else:
            self.controller = PIDController(self.blackboard)

        # --- Bộ não AI ---
        rospy.loginfo("[Init] 5. Khởi tạo AI Decision Engine...")
        self.ai = AIDecisionEngine()
        if turn_priority:
            self.ai.set_turn_priority(turn_priority)
            rospy.loginfo(f"[Init]    Turn Priority: {turn_priority}")

        # --- Bộ nhận diện biển báo và đèn giao thông ---
        rospy.loginfo("[Init] 5b. Khởi tạo Traffic Detector...")
        self.traffic_detector = TrafficDetector(
            image_width=settings.IMAGE_WIDTH,
            image_height=settings.IMAGE_HEIGHT
        )

        # --- Khởi tạo phần cứng ---
        rospy.loginfo("[Init] 6. Gọi initialize() trên phần cứng...")
        self.controller.initialize()
        rospy.loginfo("[Init]    -> Controller OK")
        self.camera.initialize()
        rospy.loginfo("[Init]    -> Camera OK")
        self.lidar.initialize()
        rospy.loginfo("[Init]    -> Lidar OK")

        # --- ROS Subscribers ---
        rospy.loginfo("[Init] 7. Đăng ký ROS Subscribers...")
        rospy.Subscriber(settings.ROS_TOPIC_LIDAR, LaserScan, self._lidar_callback)
        rospy.Subscriber(settings.ROS_TOPIC_CAMERA, Image, self._camera_callback)

        rospy.loginfo("[Init] ====== HOÀN TẤT - Node đã sẵn sàng! ======")

    # ----------------------------------------------------------
    # Callbacks: Thêm log debug để kiểm tra có nhận data không
    # ----------------------------------------------------------
    def _camera_callback(self, msg):
        rospy.logdebug("[CB] Camera: ĐÃ NHẬN frame ảnh từ ROS!")
        self.camera.ros_callback(msg)

    def _lidar_callback(self, msg):
        rospy.logdebug("[CB] Lidar:  ĐÃ NHẬN dữ liệu quét Laser từ ROS!")
        self.lidar.ros_callback(msg)

    # ----------------------------------------------------------
    # Hàm thực thi lệnh cuối cùng ra phần cứng
    # (Đây là nơi AI quyết định override hay giữ nguyên lệnh)
    # ----------------------------------------------------------
    def _execute_command(self, ai_action, ai_steer, ai_throttle):
        """Thực thi lệnh điều khiển từ AI xuống phần cứng."""
        if ai_action == Action.WAIT_RED_LIGHT:
            # Đèn đỏ -> dừng hẳn
            self.controller.stop()
        elif ai_action in (Action.TURN_LEFT, Action.TURN_RIGHT, Action.GO_STRAIGHT):
            # AI điều khiển trực tiếp (override controller) khi rẽ/đi thẳng qua ngã tư
            self.controller.move(ai_throttle, ai_steer)
        else:
            # FOLLOW_LANE: dùng góc lái từ Controller PID/Predictive, tốc độ từ AI
            steer_from_controller = self.blackboard.get('steering', 0.0)
            self.controller.move(ai_throttle, steer_from_controller)

    # ----------------------------------------------------------
    # Vòng lặp chính
    # ----------------------------------------------------------
    def run(self):
        """Vòng lặp điều khiển chính chạy ở 20Hz."""
        rospy.logdebug("[Run] Bắt đầu vòng lặp điều khiển 20Hz...")
        rate = rospy.Rate(20)

        try:
            while not rospy.is_shutdown():

                # --- Kiểm tra dữ liệu đầu vào ---
                has_image = self.blackboard.has('latest_image')
                has_scan  = self.blackboard.has('latest_scan')

                if not has_image or not has_scan:
                    rospy.logwarn_throttle(2,
                        f"[Run] TẠM DỪNG - Chờ data: Camera={'OK' if has_image else 'CHƯA'} | Lidar={'OK' if has_scan else 'CHƯA'}")
                    rate.sleep()
                    continue

                # === PIPELINE XỬ LÝ ===
                self.lidar.process(self.blackboard)
                self.fsm.process(self.blackboard)
                self.camera.process(self.blackboard)
                self.traffic_detector.process(self.blackboard)
                self.controller.process(self.blackboard)
                self.ai.process(self.blackboard)

                # Lấy lệnh từ AI
                ai_action   = self.blackboard.get('ai_action', Action.FOLLOW_LANE)
                ai_steer    = self.blackboard.get('ai_steering', 0.0)
                ai_throttle = self.blackboard.get('ai_throttle', 0.0)

                # Thực thi lệnh ra phần cứng
                self._execute_command(ai_action, ai_steer, ai_throttle)

                # Debugger: ghi CSV, video và in rospy.logdebug (tất cả debug tập trung ở đây)
                self.debugger.process(self.blackboard)

                rate.sleep()

        except KeyboardInterrupt:
            rospy.logwarn("[Run] Ctrl+C nhận được. Đang dừng hệ thống...")


    def stop(self):
        """Dừng an toàn tất cả phần cứng khi thoát."""
        rospy.logdebug("[Stop] --- BẮT ĐẦU DỪNG AN TOÀN ---")
        if hasattr(self, 'controller') and self.controller:
            rospy.logdebug("[Stop] Xả ga, trả lái về 0...")
            self.controller.stop()
        if hasattr(self, 'debugger') and self.debugger:
            rospy.logdebug("[Stop] Đóng cửa sổ debug...")
            self.debugger.close()
        rospy.logdebug("[Stop] --- ĐÃ DỪNG AN TOÀN ---")


# ============================================================
# Entry Point
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='ROS AI Navigation Node')
    parser.add_argument(
        '--turn-priority', nargs='+',
        metavar='DIRECTION',
        default=['left', 'right', 'straight'],
        help='Thứ tự ưu tiên rẽ tại ngã tư. Ví dụ: --turn-priority right left straight'
    )
    # Bỏ qua args của ROS để tránh xung đột
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from error_logger import log_crash

    args = parse_args()
    node = None
    try:
        rospy.loginfo(f"[Main] Turn Priority: {args.turn_priority}")
        node = ROSAINavigationNode(turn_priority=args.turn_priority)
        node.run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
    except Exception as e:
        log_crash("ros_ai_navigation", e)
        raise
    finally:
        if node:
            node.stop()

