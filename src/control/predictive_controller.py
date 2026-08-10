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
        # Lấy quỹ đạo mục tiêu (target path) đã được tính toán từ camera_processor
        # (lane_waypoints đã bao gồm offset an toàn nếu ở Mode B, hoặc bám nét đứt ở Mode A)
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
        # Lấy mẫu lại khoảng 8 điểm (giống thuật toán gốc) để tránh việc các điểm bị nén đặc ở xa
        # làm sai lệch phương trình parabol do hiệu ứng phối cảnh (perspective)
        if len(waypoints) > 10:
            step = len(waypoints) // 8
            sampled_waypoints = waypoints[::step]
        else:
            sampled_waypoints = waypoints
            
        ys = [pt[1] for pt in sampled_waypoints]
        xs = [pt[0] for pt in sampled_waypoints]
        try:
            # Hồi quy đa thức bậc 2: x = a*y^2 + b*y + c
            poly_coeff = np.polyfit(ys, xs, 2)
            a, b, c = poly_coeff
            
            # Chọn điểm nhìn xa (Lookahead point)
            # Trước đây 8 điểm thì chọn node 6 từ dưới lên (index 5)
            # Bây giờ có 60 điểm, ta chọn điểm tương đương về khoảng cách vật lý (index 42)
            sorted_waypoints = sorted(waypoints, key=lambda pt: pt[1], reverse=True) # Sắp xếp y giảm dần (từ gần xe ra xa)
            if len(sorted_waypoints) >= 43:
                target_pt = sorted_waypoints[42]
            else:
                target_pt = sorted_waypoints[-1] if sorted_waypoints else (settings.IMAGE_CENTER_X, 240)
            
            lookahead_y = target_pt[1]
            # Thay vì dùng phương trình parabol (x = a*y^2+b*y+c) để tính predicted_x dễ bị sai số do phối cảnh,
            # ta dùng luôn toạ độ x thực tế của điểm trên đường màu vàng/cam để nhắm bắn cực chuẩn.
            predicted_x = target_pt[0]
            
            # GIỚI HẠN: Nếu điểm nhìn xa nằm ngoài ranh giới đường quét được ở lookahead_y
            road_boundaries = blackboard.get('road_boundaries', {})
            if lookahead_y in road_boundaries:
                left_b, right_b = road_boundaries[lookahead_y]
                margin = 30 
                min_safe_x = left_b + margin
                max_safe_x = right_b - margin
                predicted_x = max(min_safe_x, min(max_safe_x, predicted_x))
            
            # CHỐI BỎ NHẢY ĐỘT NGỘT: Nếu độ lệch so với khung hình trước quá lớn
            if abs(predicted_x - self.last_predicted_x) > 60:
                # Dùng lại dự đoán liền trước và ép vào khoảng an toàn ở trung tâm
                safe_min = settings.IMAGE_CENTER_X - 45
                safe_max = settings.IMAGE_CENTER_X + 45
                predicted_x = max(safe_min, min(safe_max, self.last_predicted_x))
                
            self.last_predicted_x = predicted_x
            
            # --- TÍNH TOÁN ĐỘ CONG VÀ GÓC HƯỚNG ĐƯỜNG CONG ---
            # Đạo hàm bậc 1: dx/dy = 2*a*y + b
            dx_dy = 2 * a * lookahead_y + b
            
            # Góc hướng (Heading Angle) tính bằng radian (arctan của hệ số góc)
            heading_angle = np.arctan(dx_dy)
            
            # Độ cong (Curvature): kappa = |2*a| / (1 + (2*a*y + b)^2)^1.5
            curvature = abs(2 * a) / ((1 + dx_dy**2)**1.5)
            
            # --- CÔNG THỨC ĐIỀU KHIỂN HỢP NHẤT ---
            # 1. Tính toán sai số khoảng cách (Offset Error)
            offset_px = predicted_x - settings.IMAGE_CENTER_X
            normalized_offset = offset_px / (settings.IMAGE_WIDTH / 2.0)
            
            # 2. Phương trình lái phối hợp (Lệch làn + Hướng độ cong)
            # Kp kiểm soát kéo xe về tâm, Kd (hoặc hệ số hướng) giúp xe chuẩn bị bẻ lái theo độ cong của cua trước
            k_heading = 1.2 # Hệ số nhạy theo độ cong của đường
            steering = (self.kp * normalized_offset) - (k_heading * heading_angle)
            steering = max(settings.MIN_STEERING, min(settings.MAX_STEERING, steering))
            
            # 3. Phương trình tốc độ (Throttle) tự động giảm ga khi vào cua gắt
            # Cua càng gắt (curvature lớn hoặc heading_angle lớn), xe sẽ tự động chạy chậm lại để tránh trượt bánh
            speed_reduction = 0.6 * abs(heading_angle) # Giảm tối đa 60% tốc độ cơ bản khi cua cực gắt
            throttle = settings.BASE_SPEED * (1.0 - speed_reduction)
            throttle = max(0.12, min(settings.MAX_THROTTLE, throttle)) # Giữ tốc độ tối thiểu để không bị kẹt động cơ
            # Lưu điểm điều khiển thực tế lên blackboard phục vụ debug
            blackboard.set('lookahead_point', (int(predicted_x), lookahead_y))
            
            # Sinh ra các điểm trên đường cong để phục vụ Debug (Quét sát từ 240 đến 300)
            curve_points = []
            for y_val in range(240, 300, 10):
                x_val = int(a * (y_val**2) + b * y_val + c)
                
                # Ép đường vẽ debug chui vào lòng đường sử dụng biên ở độ cao y gần nhất
                if road_boundaries:
                    closest_y = min(road_boundaries.keys(), key=lambda k: abs(k - y_val))
                    l_b, r_b = road_boundaries[closest_y]
                    x_val = max(l_b + 30, min(r_b - 30, x_val))
                    
                curve_points.append((x_val, y_val))
                
            self.move(throttle, steering)
            blackboard.set('steering', steering)
            blackboard.set('throttle', throttle)
            blackboard.set('predicted_curve', curve_points)
            
        except Exception as e:
            print(f"[PredictiveController] Lỗi polyfit hoặc tính toán curvature: {e}")
            # Fallback
            center_x = blackboard.get('center_x', settings.IMAGE_CENTER_X)
            offset_px = center_x - settings.IMAGE_CENTER_X
            normalized_offset = offset_px / (settings.IMAGE_WIDTH / 2.0)
            steering = self.kp * normalized_offset
            steering = max(settings.MIN_STEERING, min(settings.MAX_STEERING, steering))
            
            blackboard.set('lookahead_point', (int(center_x), 240))
            self.move(settings.BASE_SPEED, steering)
            blackboard.set('steering', steering)
            blackboard.set('throttle', settings.BASE_SPEED)
            blackboard.set('predicted_curve', [])

