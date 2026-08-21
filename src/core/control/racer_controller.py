# -*- coding: utf-8 -*-
"""
RacerController: Giao tiếp phần cứng điều khiển xe JetRacer Pro / JetBot Pro
"""

import rospy

class RacerController:
    def __init__(self):
        self.car = None
        self._mock = False
        self.initialize_hardware()

    def initialize_hardware(self):
        """Khởi tạo JetRacer Pro hoặc JetBot Pro fallback."""
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self.car = NvidiaRacecar()
            self.car.steering = 0.0
            self.car.throttle = 0.0
            rospy.loginfo("[HARDWARE] Khởi tạo NvidiaRacecar (JetRacer Pro) THÀNH CÔNG!")
            return
        except Exception as e:
            rospy.logwarn(f"[HARDWARE] Không tìm thấy jetracer: {e}")

        try:
            from jetbot import Robot
            self.car = Robot()
            rospy.loginfo("[HARDWARE] Khởi tạo Robot (JetBot Pro fallback) THÀNH CÔNG!")
            return
        except Exception as e:
            rospy.logwarn(f"[HARDWARE] Không tìm thấy jetbot: {e}")

        rospy.logwarn("[HARDWARE] Không tìm thấy phần cứng thật -> Chạy ở chế độ MÔ PHỎNG (Mock).")
        from unittest.mock import Mock
        self.car = Mock()
        self._mock = True

    def set_steering(self, steering):
        """Đặt góc lái [-1.0, 1.0]."""
        steering_clamped = max(-1.0, min(1.0, steering))
        if hasattr(self.car, 'steering'):
            self.car.steering = steering_clamped

    def set_throttle(self, throttle):
        """Đặt tốc độ ga [0.0, 1.0]."""
        throttle_clamped = max(0.0, min(1.0, throttle))
        if hasattr(self.car, 'throttle'):
            self.car.throttle = throttle_clamped

    def stop(self):
        """Dừng xe khẩn cấp."""
        self.set_steering(0.0)
        self.set_throttle(0.0)
