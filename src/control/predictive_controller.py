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
        # Tái sử dụng hệ số P, vì đây thực chất là Proportional control trên điểm lookahead
        self.kp = settings.PID_KP 
        self.car = None
        self._mock = False
        
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
        waypoints = blackboard.get('lane_waypoints', [])
        
        if not waypoints or len(waypoints) < 2:
            # Fallback nếu không có đủ điểm
            center_x = blackboard.get('center_x', settings.IMAGE_CENTER_X)
            offset_px = center_x - settings.IMAGE_CENTER_X
            normalized_offset = offset_px / (settings.IMAGE_WIDTH / 2.0)
            steering = self.kp * normalized_offset
            
            # Giới hạn góc lái
            steering = max(settings.MIN_STEERING, min(settings.MAX_STEERING, steering))
            
            # Lưu điểm điều khiển giả định
            blackboard.set('lookahead_point', (int(center_x), 240))
            
            self.move(settings.BASE_SPEED, steering)
            blackboard.set('steering', steering)
            blackboard.set('predicted_curve', [])
            return

        # Hồi quy đa thức bậc 2: x = a*y^2 + b*y + c
        # Fit x theo y vì y tăng đều đặn từ trên xuống dưới ảnh
        ys = [pt[1] for pt in waypoints]
        xs = [pt[0] for pt in waypoints]
        
        try:
            poly_coeff = np.polyfit(ys, xs, 2)
            
            # Chọn điểm nhìn xa (Lookahead point). y càng nhỏ nghĩa là càng xa về phía đỉnh ảnh.
            lookahead_y = 160 
            predicted_x = np.polyval(poly_coeff, lookahead_y)
            
            # Tính toán offset từ tâm ảnh tới điểm dự đoán (predicted_x - CENTER)
            offset_px = predicted_x - settings.IMAGE_CENTER_X
            
            # Chuẩn hóa offset về khoảng [-1, 1]
            normalized_offset = offset_px / (settings.IMAGE_WIDTH / 2.0)
            
            # Tính góc lái
            steering = self.kp * normalized_offset
            
            # Giới hạn góc lái
            steering = max(settings.MIN_STEERING, min(settings.MAX_STEERING, steering))
            
            # Lưu điểm điều khiển thực tế
            blackboard.set('lookahead_point', (int(predicted_x), lookahead_y))
            
            # Sinh ra các điểm trên đường cong để phục vụ Debug/Vẽ đồ thị
            curve_points = []
            for y_val in range(160, 300, 20):
                x_val = int(np.polyval(poly_coeff, y_val))
                curve_points.append((x_val, y_val))
                
            self.move(settings.BASE_SPEED, steering)
            blackboard.set('steering', steering)
            blackboard.set('predicted_curve', curve_points)
            
        except Exception as e:
            print(f"[PredictiveController] Lỗi polyfit: {e}")
            # Fallback
            center_x = blackboard.get('center_x', settings.IMAGE_CENTER_X)
            offset_px = center_x - settings.IMAGE_CENTER_X
            normalized_offset = offset_px / (settings.IMAGE_WIDTH / 2.0)
            steering = self.kp * normalized_offset
            steering = max(settings.MIN_STEERING, min(settings.MAX_STEERING, steering))
            
            # Lưu điểm điều khiển giả định
            blackboard.set('lookahead_point', (int(center_x), 240))
            
            self.move(settings.BASE_SPEED, steering)
            blackboard.set('steering', steering)
            blackboard.set('predicted_curve', [])

