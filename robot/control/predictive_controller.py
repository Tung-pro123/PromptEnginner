import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from control.base_controller import BaseController
from config import settings

class PredictiveController(BaseController):
    def __init__(self, blackboard=None):
        self.blackboard = blackboard
        # Tăng mạnh độ nhạy lái của bộ điều khiển Predictive (nhân 6.0)
        self.kp = settings.PID_KP * 6.0
        self.car = None
        self._mock = False
        self.last_predicted_x = settings.IMAGE_CENTER_X
        
    def initialize(self):
        """Khởi tạo phần cứng JetRacer hoặc chế độ Mock."""
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self.car = NvidiaRacecar()
            self.car.steering = 0.0
            self.car.throttle = 0.0
            print("[INFO] Khởi tạo JetRacer (NvidiaRacecar) thành công cho PredictiveController.")
            return
        except Exception as e:
            print(f"[WARN] Không tìm thấy thư viện jetracer: {e}")

        # try:
        #     from jetbot import Robot
        #     self.car = Robot()
        #     self._mock = False
        #     print("[INFO] Khởi tạo JetBot Pro (fallback) thành công cho PredictiveController.")
        #     return
        # except Exception as e:
        #     print(f"[WARN] Không tìm thấy thư viện jetbot: {e}")

        # print("[WARN] Không tìm thấy phần cứng → Chạy ở chế độ MÔ PHỎNG (Mock).")
        # from unittest.mock import Mock
        # self.car = Mock()
        # self._mock = True

    def move(self, speed, direction):
        """Thực thi lệnh lái và ga xuống phần cứng."""
        self._set_steering(direction)
        self._set_throttle(speed)

    def stop(self):
        """Dừng xe khẩn cấp."""
        self._set_throttle(0.0)
        self._set_steering(0.0)

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

    def process(self, blackboard):
        # Lấy tọa độ mục tiêu từ Hybrid Lane Detector (đã được camera_processor tính toán)
        target_x = blackboard.get('target_x', settings.IMAGE_CENTER_X)
        left_border = blackboard.get('left_border', 0)
        right_border = blackboard.get('right_border', settings.IMAGE_WIDTH - 1)
        
        # Dùng target_x để điều hướng.
        # Có thể thêm logic kẹp (clamp) `target_x` nằm gọn giữa left_border và right_border nếu cần.
        predicted_x = target_x
        
        # CHỐI BỎ NHẢY ĐỘT NGỘT: Nếu độ lệch so với khung hình trước quá lớn
        if abs(predicted_x - self.last_predicted_x) > 60:
            # Dùng lại dự đoán liền trước và ép vào khoảng an toàn ở trung tâm
            safe_min = settings.IMAGE_CENTER_X - 45
            safe_max = settings.IMAGE_CENTER_X + 45
            predicted_x = max(safe_min, min(safe_max, self.last_predicted_x))
            
        self.last_predicted_x = predicted_x
        
        # 1. Tính toán sai số khoảng cách (Offset Error)
        offset_px = predicted_x - settings.IMAGE_CENTER_X
        normalized_offset = offset_px / (settings.IMAGE_WIDTH / 2.0)
        
        # 2. Tính toán Steering (Góc lái) dựa trên KP
        steering = self.kp * normalized_offset
        steering = max(settings.MIN_STEERING, min(settings.MAX_STEERING, steering))
        
        # 3. Phương trình tốc độ (Throttle) tự động giảm ga khi vào cua gắt
        # Cua càng gắt (offset lớn), xe sẽ tự động chạy chậm lại
        speed_reduction = 0.4 * abs(normalized_offset) # Giảm tối đa 40%
        throttle = settings.BASE_SPEED * (1.0 - speed_reduction)
        throttle = max(0.12, min(settings.MAX_THROTTLE, throttle))
        
        # Lưu điểm điều khiển thực tế lên blackboard phục vụ debug
        # Lấy Y của Lookahead ROI = 55% height = 0.55 * 300 = 165
        blackboard.set('lookahead_point', (int(predicted_x), int(settings.IMAGE_HEIGHT * 0.55)))
        
        # Thực thi lệnh
        self.move(throttle, steering)
        blackboard.set('steering', steering)
        blackboard.set('throttle', throttle)
        blackboard.set('predicted_curve', []) # Không còn đường cong polyfit

