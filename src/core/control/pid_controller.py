#!/usr/bin/env python3
"""
PID Controller cho JetRacer Speed Track.
Áp dụng từ bài báo s43684-021-00015-x: Hàm hiệu năng năng lượng (Energy Efficiency)
kết hợp PID để điều khiển bám lane mượt mà và ổn định.
"""

import time


class PIDController:
    """Bộ điều khiển PID chuẩn với anti-windup và derivative filter.
    
    Tham chiếu bài báo (Section 2.2): Hàm đánh giá hiệu suất U1 dựa trên
    tốc độ thực tế vs tốc độ mong muốn → PID giúp giảm thiểu sai số này.
    """

    def __init__(self, kp=0.5, ki=0.0, kd=0.1, 
                 output_min=-1.0, output_max=1.0,
                 integral_limit=0.5):
        """
        Args:
            kp: Hệ số tỷ lệ (Proportional gain)
            ki: Hệ số tích phân (Integral gain) 
            kd: Hệ số vi phân (Derivative gain)
            output_min: Giới hạn dưới đầu ra
            output_max: Giới hạn trên đầu ra
            integral_limit: Giới hạn anti-windup cho thành phần tích phân
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit

        # Trạng thái nội bộ
        self._prev_error = 0.0
        self._integral = 0.0
        self._last_time = None

    def reset(self):
        """Reset trạng thái PID (khi chuyển trạng thái FSM)."""
        self._prev_error = 0.0
        self._integral = 0.0
        self._last_time = None

    def compute(self, error, current_time=None):
        """Tính toán đầu ra PID.
        
        Args:
            error: Sai số hiện tại (ví dụ: lane_center - image_center)
            current_time: Thời gian hiện tại (seconds). Nếu None, dùng time.time().
            
        Returns:
            float: Giá trị điều chỉnh (đã được clamp trong [output_min, output_max])
        """
        if current_time is None:
            current_time = time.time()

        # Tính dt
        if self._last_time is None:
            dt = 0.05  # Giá trị mặc định cho lần đầu (~20 FPS)
        else:
            dt = current_time - self._last_time
            if dt <= 0:
                dt = 0.05

        # Thành phần tỷ lệ (P)
        p_term = self.kp * error

        # Thành phần tích phân (I) với anti-windup
        self._integral += error * dt
        self._integral = max(-self.integral_limit, 
                            min(self.integral_limit, self._integral))
        i_term = self.ki * self._integral

        # Thành phần vi phân (D) 
        d_term = self.kd * (error - self._prev_error) / dt

        # Tổng đầu ra
        output = p_term + i_term + d_term

        # Clamp đầu ra
        output = max(self.output_min, min(self.output_max, output))

        # Cập nhật trạng thái
        self._prev_error = error
        self._last_time = current_time

        return output
