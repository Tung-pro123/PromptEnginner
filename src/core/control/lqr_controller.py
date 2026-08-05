#!/usr/bin/env python3
"""
LQR Controller & Path Planning Helpers for JetRacer Pro
Hỗ trợ bám làn đường thẳng/cong, né vật cản theo phương pháp dịch vạch ảo (Offset) 
và chui cổng căn giữa tự động.

Nhập khẩu:
    from src.core.control.lqr_controller import LQRController, ObstacleDetector
"""

import sys
# Lọc bỏ đường dẫn python2.7 của ROS Melodic để không đè lên các thư viện Python 3
sys.path = [p for p in sys.path if 'python2.7' not in p]

import numpy as np
import math
import time

class LQRController:
    """
    Bộ điều khiển tối ưu LQR dựa trên mô hình xe đạp động học (Kinematic Bicycle Model).
    """
    def __init__(self, wheelbase=0.18, scale_factor=0.0015):
        """
        Args:
            wheelbase: Chiều dài cơ sở xe (khoảng cách 2 trục bánh xe), JetRacer Pro ~ 0.18m - 0.20m
            scale_factor: Hệ số đổi từ pixel sang mét thực tế (ví dụ: 1 pixel ~ 1.5mm)
        """
        self.L = wheelbase
        self.scale_factor = scale_factor
        
        # Ma trận Q phạt sai số trạng thái [e, e_dot, e_theta, e_theta_dot]
        # e: sai số khoảng cách, e_theta: sai số góc
        self.Q = np.diag([15.0, 1.0, 8.0, 0.5]) 
        # Ma trận R phạt nỗ lực điều khiển góc lái
        self.R = np.array([[1.2]])
        
        # Lưu các trạng thái cũ để tính đạo hàm (rate of change)
        self.last_e = 0.0
        self.last_e_theta = 0.0
        self.last_time = time.time()
        
        # Biến điều khiển dịch vạch né vật cản (Offset)
        self.target_offset = 0.0
        self.current_offset = 0.0
        self.ramp_speed = 0.35  # Tốc độ dịch vạch tiếp tuyến (m/s)

    def solve_DARE(self, A, B, Q, R):
        """Giải phương trình Riccati đại số rời rạc để tìm ma trận hồi tiếp K tối ưu."""
        P = Q.copy()
        for _ in range(100):
            # Công thức Riccati Iteration
            P_next = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
            if np.allclose(P, P_next, rtol=1e-5, atol=1e-5):
                break
            P = P_next
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def update_offset(self, dt):
        """Tăng/giảm dần offset thực tế (current_offset) về phía offset mục tiêu (target_offset)
        để tạo đường chuyển làn S-curve tiếp tuyến mượt mà."""
        diff = self.target_offset - self.current_offset
        if abs(diff) > 0.001:
            step = np.sign(diff) * self.ramp_speed * dt
            if abs(step) > abs(diff):
                self.current_offset = self.target_offset
            else:
                self.current_offset += step
        else:
            self.current_offset = self.target_offset

    def compute_steering(self, C_near, C_far, Y_near, Y_far, speed, image_width=300, auto_ramp=False):
        """
        Tính toán góc đánh lái tối ưu bằng LQR dựa trên thông tin đường từ Camera.
        
        Args:
            C_near: Tọa độ X của tâm đường ở vùng ROI gần (0 -> image_width)
            C_far: Tọa độ X của tâm đường ở vùng ROI xa (0 -> image_width)
            Y_near: Tọa độ dòng Y của ROI gần (ví dụ 260)
            Y_far: Tọa độ dòng Y của ROI xa (ví dụ 140)
            speed: Tốc độ hiện tại của xe (có thể là throttle 0->1.0 hoặc m/s)
            image_width: Chiều rộng ảnh camera (mặc định 300px)
            auto_ramp: Nếu True, LQR tự ramp offset. Nếu False, nhận current_offset trực tiếp từ FSM.
        """
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.05
        self.last_time = current_time

        # 1. Cập nhật đường dịch vạch ảo né vật cản nếu bật auto_ramp
        if auto_ramp:
            self.update_offset(dt)

        # 2. Tính sai số khoảng cách (e) quy ra mét với DẤU CHUẨN VẬT LÝ (cộng offset):
        # Khi xe lách sang phải (offset > 0), vạch đỏ sẽ bị trôi sang trái trong ảnh (C_near < image_width/2 -> e_pixel < 0).
        # Do đó e = e_pixel * scale + current_offset sẽ đưa sai số tổng về 0 tại vị trí né mới.
        e_pixel = C_near - (image_width / 2.0)
        e = e_pixel * self.scale_factor + self.current_offset

        # 3. Tính sai số góc hướng (e_theta) quy ra radian
        dx = (C_far - C_near) * self.scale_factor
        dy = (Y_near - Y_far) * self.scale_factor
        e_theta = math.atan2(dx, dy)

        # 4. Tính đạo hàm của sai số
        e_dot = (e - self.last_e) / dt
        e_theta_dot = (e_theta - self.last_e_theta) / dt

        self.last_e = e
        self.last_e_theta = e_theta

        # 5. Nếu xe dừng hẳn thì không đánh lái
        if speed < 0.05:
            return 0.0

        # Ước tính tốc độ thực (m/s): Nếu speed <= 1.0 (throttle), quy đổi sang m/s (JetRacer max speed ~1.5 m/s)
        v_ms = speed if speed > 1.0 else max(0.1, speed * 1.5)

        # 6. Xây dựng mô hình không gian trạng thái của xe (State-Space matrices) dựa trên v_ms (m/s)
        A = np.array([
            [1.0, dt, 0.0, 0.0],
            [0.0, 0.0, v_ms, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0]
        ])
        B = np.array([[0.0], [0.0], [v_ms / self.L], [0.0]])

        # 7. Tính ma trận K tối ưu bằng LQR
        try:
            K = self.solve_DARE(A, B, self.Q, self.R)
        except Exception:
            # Fallback nếu lỗi ma trận
            return -0.5 * (e / self.scale_factor) / (image_width / 2.0)

        # 8. Tính góc bẻ lái u = -K * x
        x = np.array([[e], [e_dot], [e_theta], [e_theta_dot]])
        steering_rad = -(K @ x)[0, 0]

        # 9. Quy đổi từ radian góc lái thực tế sang tỷ lệ điều khiển của JetRacer [-1.0, 1.0]
        max_steering_rad = math.radians(30)
        steering_command = np.clip(steering_rad / max_steering_rad, -1.0, 1.0)

        return steering_command


class ObstacleDetector:
    """
    Bộ giám sát khoảng cách vật cản phía trước bằng cảm biến LiDAR (RPLIDAR).
    """
    def __init__(self, trigger_distance_base=0.45, reaction_time=0.5, safe_distance=0.20):
        """
        Args:
            trigger_distance_base: Khoảng cách kích hoạt mặc định (mét)
            reaction_time: Thời gian hệ thống phản ứng (giây)
            safe_distance: Khoảng cách an toàn hình học (mét)
        """
        self.base_dist = trigger_distance_base
        self.reaction_time = reaction_time
        self.safe_dist = safe_distance

    def get_trigger_distance(self, speed):
        """Tính toán khoảng cách kích hoạt né tối ưu dựa trên tốc độ thực tế của xe."""
        return max(0.35, speed * self.reaction_time + self.safe_dist)

    def get_front_obstacle_distance(self, scan_msg, angle_range_deg=15.0):
        """
        Quét các tia LiDAR ở góc trước mặt xe để tìm khoảng cách tới vật cản gần nhất.
        
        Args:
            scan_msg: ROS LaserScan message
            angle_range_deg: Góc quét trước mặt (-15 độ tới +15 độ)
        """
        if scan_msg is None or not hasattr(scan_msg, 'ranges'):
            return float('inf')

        front_distances = []
        for i, dist in enumerate(scan_msg.ranges):
            # Tính toán góc của tia quét hiện tại
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            angle_deg = math.degrees(angle)
            
            # Bù 180 độ do góc xoay lắp đặt LiDAR ngược trên xe JetRacer
            angle_deg = angle_deg + 180.0
            # Chuẩn hóa góc về [-180, 180]
            angle_deg = (angle_deg + 180) % 360 - 180
            
            # Nếu tia nằm trong góc quét trước mặt xe
            if abs(angle_deg) <= angle_range_deg:
                if scan_msg.range_min < dist < scan_msg.range_max:
                    front_distances.append(dist)
                    
        if front_distances:
            return min(front_distances)
        return float('inf')
