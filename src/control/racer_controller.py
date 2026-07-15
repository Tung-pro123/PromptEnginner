import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control.base_controller import BaseController
from config import settings

class RacerController(BaseController):
    """
    Controller cho JetRacer tích hợp thuật toán PID điều khiển lái.
    Kế thừa từ BaseController.
    """
    def __init__(self):
        self.car = None
        self._mock = False
        
        # PID state
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = time.time()
        
    def initialize(self):
        """Khởi tạo phần cứng JetRacer hoặc chế độ Mock."""
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self.car = NvidiaRacecar()
            self.car.steering = 0.0
            self.car.throttle = 0.0
            print("[INFO] Khởi tạo JetRacer (NvidiaRacecar) thành công.")
            return
        except Exception as e:
            print(f"[WARN] Không tìm thấy thư viện jetracer: {e}")

        try:
            from jetbot import Robot
            self.car = Robot()
            self._mock = False
            print("[INFO] Khởi tạo JetBot Pro (fallback) thành công.")
            return
        except Exception as e:
            print(f"[WARN] Không tìm thấy thư viện jetbot: {e}")

        print("[WARN] Không tìm thấy phần cứng → Chạy ở chế độ MÔ PHỎNG (Mock).")
        from unittest.mock import Mock
        self.car = Mock()
        self._mock = True

    def calculate_steering(self, center_x, current_offset_px):
        """
        Tính toán góc lái dựa trên PID Controller (Bám line).
        Hàm này tương tự `correct_course_pid` ở bản cũ.
        """
        error = (center_x + current_offset_px) - settings.IMAGE_CENTER_X
        # Chuẩn hóa lỗi về đoạn [-1, 1]
        normalized_error = error / (settings.IMAGE_WIDTH / 2.0)
        
        # Kiểm tra vùng an toàn (dead zone)
        if abs(normalized_error) < settings.SAFE_ZONE_PERCENT:
            return 0.0
            
        current_time = time.time()
        dt = current_time - self._pid_last_time
        if dt <= 0:
            dt = 0.05
            
        # P - Proportional
        p_term = settings.PID_KP * normalized_error
        
        # I - Integral (Anti-windup)
        self._pid_integral += normalized_error * dt
        self._pid_integral = max(-1.0, min(1.0, self._pid_integral))
        i_term = settings.PID_KI * self._pid_integral
        
        # D - Derivative
        d_error = (normalized_error - self._pid_last_error) / dt
        d_term = settings.PID_KD * d_error
        
        steering_output = p_term + i_term + d_term
        
        # Lưu trạng thái PID
        self._pid_last_error = normalized_error
        self._pid_last_time = current_time
        
        return max(settings.MIN_STEERING, min(settings.MAX_STEERING, steering_output))

    def reset_pid(self):
        """Reset các giá trị PID khi chuyển trạng thái (như bắt đầu DODGING)."""
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = time.time()

    def move(self, speed, direction):
        """Thực thi lệnh lái và ga xuống phần cứng."""
        self._set_steering(direction)
        self._set_throttle(speed)

    def stop(self):
        """Dừng xe khẩn cấp."""
        self._set_throttle(0.0)
        self._set_steering(0.0)
        
    def turn_angle(self, degrees):
        """Rẽ một góc cho trước (vòng cung)."""
        if degrees == 0:
            return
            
        duration = abs(degrees) / 90.0 * 1.5 # Giả sử 1.5s cho 90 độ
        turn_steering = 0.7 if degrees > 0 else -0.7
        turn_throttle = settings.BASE_SPEED * 0.8 # Đi chậm lại khi rẽ
        
        self.move(turn_throttle, turn_steering)
        time.sleep(duration)
        self.stop()

    # --- Internal Helpers ---
    def _set_throttle(self, value):
        value = max(-settings.MAX_THROTTLE, min(settings.MAX_THROTTLE, value))
        if hasattr(self.car, 'throttle'):
            self.car.throttle = value
        elif hasattr(self.car, 'set_motors'):
            self.car.set_motors(value, value)

    def _set_steering(self, value):
        value = max(settings.MIN_STEERING, min(settings.MAX_STEERING, value))
        value += settings.STEERING_OFFSET
        if hasattr(self.car, 'steering'):
            self.car.steering = value
