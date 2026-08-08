# -*- coding: utf-8 -*-
"""
LQRController: Bộ điều khiển tối ưu LQR dựa trên mô hình Xe đạp Động học (Kinematic Bicycle Model).
Giải phương trình Riccati đại số rời rạc (DARE) để dập tắt dao động bẻ lái.
"""

import numpy as np
import math
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class LQRController:
    def __init__(self, blackboard=None):
        self.blackboard = blackboard
        self.L = settings.LQR_WHEELBASE
        self.scale_factor = settings.LQR_SCALE_FACTOR
        
        self.Q = np.diag(settings.LQR_Q_DIAG)
        self.R = np.array(settings.LQR_R_VAL)
        
        self.last_e = 0.0
        self.last_e_theta = 0.0
        self.last_time = time.time()

    def solve_DARE(self, A, B, Q, R):
        """Giải phương trình Riccati đại số rời rạc để tìm ma trận Gain hồi tiếp K tối ưu."""
        P = Q.copy()
        for _ in range(100):
            P_next = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
            if np.allclose(P, P_next, rtol=1e-5, atol=1e-5):
                break
            P = P_next
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def compute_steering(self, C_near, C_far, Y_near, Y_far, speed, current_offset_px):
        """Tính toán góc lái LQR từ 2 điểm C_near, C_far."""
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: dt = 0.05
        self.last_time = now

        e_pixel = C_near - settings.IMAGE_CENTER_X
        e = e_pixel * self.scale_factor + (current_offset_px * self.scale_factor)
        e_dot = (e - self.last_e) / dt
        self.last_e = e

        dx = C_far - C_near
        dy = Y_near - Y_far
        e_theta = math.atan2(dx, dy)
        e_theta_dot = (e_theta - self.last_e_theta) / dt
        self.last_e_theta = e_theta

        v = max(0.1, speed)
        A = np.array([
            [1.0,  dt,   0.0, 0.0],
            [0.0, 0.0,     v, 0.0],
            [0.0, 0.0,   1.0,  dt],
            [0.0, 0.0,   0.0, 0.0]
        ])
        B = np.array([
            [0.0],
            [0.0],
            [0.0],
            [v / self.L]
        ])

        try:
            K = self.solve_DARE(A, B, self.Q, self.R)
            x_state = np.array([[e], [e_dot], [e_theta], [e_theta_dot]])
            u = -K @ x_state
            steering = float(u[0, 0])
        except Exception:
            steering = e_pixel * settings.KP

        return max(settings.MIN_STEERING, min(settings.MAX_STEERING, steering))
