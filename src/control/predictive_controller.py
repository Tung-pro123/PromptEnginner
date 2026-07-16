import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from control.base_controller import BaseController
from config import settings

class PredictiveController(BaseController):
    def __init__(self, blackboard=None):
        self.blackboard = blackboard
        # Tái sử dụng hệ số P, vì đây thực chất là Proportional control trên điểm lookahead
        self.kp = settings.PID_P 
        
    def process(self, blackboard):
        waypoints = blackboard.get('lane_waypoints', [])
        
        if not waypoints or len(waypoints) < 2:
            # Fallback nếu không có đủ điểm
            center_x = blackboard.get('center_x', settings.IMAGE_CENTER_X)
            offset_px = settings.IMAGE_CENTER_X - center_x
            steering = self.kp * offset_px
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
            
            # Tính toán offset từ tâm ảnh tới điểm dự đoán
            offset_px = settings.IMAGE_CENTER_X - predicted_x
            
            # Tính góc lái
            steering = self.kp * offset_px
            
            # Giới hạn góc lái
            steering = max(min(steering, 1.0), -1.0)
            
            # Sinh ra các điểm trên đường cong để phục vụ Debug/Vẽ đồ thị
            curve_points = []
            for y_val in range(160, 300, 20):
                x_val = int(np.polyval(poly_coeff, y_val))
                curve_points.append((x_val, y_val))
                
            blackboard.set('steering', steering)
            blackboard.set('predicted_curve', curve_points)
            
        except Exception as e:
            print(f"[PredictiveController] Lỗi polyfit: {e}")
            # Fallback
            center_x = blackboard.get('center_x', settings.IMAGE_CENTER_X)
            offset_px = settings.IMAGE_CENTER_X - center_x
            blackboard.set('steering', self.kp * offset_px)
            blackboard.set('predicted_curve', [])
