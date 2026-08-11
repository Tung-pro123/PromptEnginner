import unittest
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.control.pid_controller import PIDController
from src.config import settings

class TestPIDController(unittest.TestCase):
    def setUp(self):
        self.controller = PIDController()
        # Mock xe để không gọi xuống phần cứng thật trong lúc chạy test
        from unittest.mock import Mock
        self.controller.car = Mock()
        self.controller._mock = True

    def test_calculate_steering_pid_straight(self):
        """Kiểm tra đi thẳng (error nằm trong vùng an toàn)."""
        settings.IMAGE_CENTER_X = 150
        settings.IMAGE_WIDTH = 300
        settings.SAFE_ZONE_PERCENT = 0.3 # Vùng an toàn 30%
        
        # Xe đang ở tâm (150)
        center_x = 150
        current_offset_px = 0
        
        steering = self.controller.calculate_steering(center_x, current_offset_px)
        self.assertEqual(steering, 0.0)

    def test_calculate_steering_pid_deviated(self):
        """Kiểm tra đánh lái khi xe lệch khỏi vùng an toàn."""
        # Use settings module that pid_controller imported
        from src.control.pid_controller import settings as pid_settings
        pid_settings.IMAGE_CENTER_X = 150
        pid_settings.IMAGE_WIDTH = 300
        pid_settings.SAFE_ZONE_PERCENT = 0.1
        pid_settings.PID_KP = 0.5
        pid_settings.PID_KI = 0.0
        pid_settings.PID_KD = 0.0
        
        # Xe lệch phải (center_x lớn)
        center_x = 250
        current_offset_px = 0
        
        # Error = 250 - 150 = 100 px. 
        # Normalized error = 100 / 150 = 0.666
        # P = 0.5 * 0.666 = 0.333
        steering = self.controller.calculate_steering(center_x, current_offset_px)
        self.assertAlmostEqual(steering, 0.333, places=2)

    def test_clamping_logic(self):
        """Kiểm tra giới hạn góc lái và tốc độ ga."""
        settings.MAX_STEERING = 1.0
        settings.MIN_STEERING = -1.0
        settings.MAX_THROTTLE = 0.4
        
        self.controller.move(speed=1.5, direction=-2.0)
        
        # car.set_motors hoặc car.throttle nên nhận giá trị đã bị clamp
        if hasattr(self.controller.car, 'throttle'):
            self.assertEqual(self.controller.car.throttle, 0.4)
            self.assertEqual(self.controller.car.steering, -1.0)
        elif hasattr(self.controller.car, 'set_motors'):
            self.controller.car.set_motors.assert_called_with(0.4, 0.4)

    def test_stop(self):
        """Kiểm tra hàm dừng khẩn cấp."""
        self.controller.stop()
        if hasattr(self.controller.car, 'throttle'):
            self.assertEqual(self.controller.car.throttle, 0.0)
            self.assertEqual(self.controller.car.steering, 0.0)
        elif hasattr(self.controller.car, 'set_motors'):
            self.controller.car.set_motors.assert_called_with(0.0, 0.0)

if __name__ == '__main__':
    unittest.main()
