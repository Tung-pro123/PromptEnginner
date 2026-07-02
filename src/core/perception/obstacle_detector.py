#!/usr/bin/env python3
"""
Module Phát hiện Vật cản (Obstacle Detector) cho Speed Track.

Áp dụng từ bài báo s43684-021-00015-x (Section 2.3):
- Mô hình khoảng cách an toàn tối thiểu Min(S) = V_A * t_d + D_0
- Đánh giá an toàn U2: so sánh ΔX (khoảng cách thực tế) với Min(S)
- Grid-based phân vùng: quét LiDAR theo sector trái/giữa/phải

Thiết kế cho sa bàn Speed Track:
- Quét vùng phía trước [-45°, +45°] để phát hiện vật cản
- Quét vùng trái/phải để quyết định hướng né
"""

import numpy as np
import math


class ObstacleDetector:
    """Phát hiện vật cản từ dữ liệu LiDAR LaserScan.
    
    Tham chiếu bài báo:
    - Section 2.3: Min_c(S) = V_A * t_d + D_0 (khoảng cách an toàn)
    - Section 2.4: U3 - Lane vacancy rate (tỷ lệ trống bên trái/phải)
    """

    def __init__(self,
                 safety_distance=0.35,
                 warning_distance=0.55,
                 min_valid_range=0.10,
                 max_valid_range=1.5,
                 front_angle_range=45.0,
                 side_angle_range=30.0,
                 min_obstacle_points=5):
        """
        Args:
            safety_distance: Khoảng cách an toàn tối thiểu Min(S) (mét).
                            Ánh xạ từ công thức Min(S) = V_A * t_d + D_0.
                            Với tốc độ JetRacer ~0.3 m/s, t_d ~0.5s, D_0 ~0.2m → ~0.35m
            warning_distance: Khoảng cách bắt đầu cảnh báo (mét)
            min_valid_range: Khoảng cách LiDAR tối thiểu hợp lệ (mét)
            max_valid_range: Khoảng cách LiDAR tối đa xem xét (mét)
            front_angle_range: Nửa góc quét phía trước (độ). ±45° = quét tổng 90°
            side_angle_range: Nửa góc quét bên trái/phải (độ)
            min_obstacle_points: Số điểm LiDAR tối thiểu để xác nhận vật cản
        """
        self.safety_distance = safety_distance
        self.warning_distance = warning_distance
        self.min_valid_range = min_valid_range
        self.max_valid_range = max_valid_range
        self.front_angle_range = front_angle_range
        self.side_angle_range = side_angle_range
        self.min_obstacle_points = min_obstacle_points

        # Kết quả phân tích gần nhất
        self.last_result = None

    def analyze(self, scan_msg):
        """Phân tích dữ liệu LiDAR LaserScan và trả về kết quả phát hiện.
        
        Áp dụng Grid-based sector scanning từ bài báo (Section 2.4, Fig. 4):
        Chia vùng quét thành 3 sector: FRONT, LEFT, RIGHT.
        
        Args:
            scan_msg: ROS LaserScan message
            
        Returns:
            dict: {
                'obstacle_detected': bool,
                'front_distance': float (khoảng cách vật cản phía trước),
                'left_clear': float (khoảng cách trống bên trái),
                'right_clear': float (khoảng cách trống bên phải),
                'avoid_direction': str ('left' | 'right' | 'none'),
                'danger_level': str ('safe' | 'warning' | 'danger')
            }
        """
        if scan_msg is None:
            return self._default_result()

        ranges = np.array(scan_msg.ranges)
        n = len(ranges)
        if n == 0:
            return self._default_result()

        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment

        # Tạo mảng góc (radians) cho mỗi điểm LiDAR
        angles_rad = angle_min + np.arange(n) * angle_increment
        angles_deg = np.degrees(angles_rad)

        # Lọc giá trị hợp lệ
        valid_mask = np.isfinite(ranges) & (ranges >= self.min_valid_range) & (ranges <= self.max_valid_range)

        # === SECTOR SCANNING (Grid-based từ bài báo Section 2.4) ===
        
        # Sector FRONT: [-front_angle_range, +front_angle_range] quanh 0°
        front_mask = valid_mask & (np.abs(angles_deg) <= self.front_angle_range)
        front_ranges = ranges[front_mask]
        
        # Sector LEFT: [front_angle_range, front_angle_range + side_angle_range*2] 
        left_start = self.front_angle_range
        left_end = self.front_angle_range + self.side_angle_range * 2
        left_mask = valid_mask & (angles_deg >= left_start) & (angles_deg <= left_end)
        left_ranges = ranges[left_mask]
        
        # Sector RIGHT: [-front_angle_range - side_angle_range*2, -front_angle_range]
        right_start = -(self.front_angle_range + self.side_angle_range * 2)
        right_end = -self.front_angle_range
        right_mask = valid_mask & (angles_deg >= right_start) & (angles_deg <= right_end)
        right_ranges = ranges[right_mask]

        # === TÍNH KHOẢNG CÁCH TỐI THIỂU TỪNG SECTOR ===
        front_distance = float(np.min(front_ranges)) if len(front_ranges) >= self.min_obstacle_points else float('inf')
        left_clearance = float(np.min(left_ranges)) if len(left_ranges) > 0 else float('inf')
        right_clearance = float(np.min(right_ranges)) if len(right_ranges) > 0 else float('inf')

        # === ĐÁNH GIÁ AN TOÀN (từ bài báo Section 2.3, Eq. 9) ===
        # U2 = Min(S) - ΔX / ΔX  nếu ΔX < Min(S)
        # U2 = 1                  nếu ΔX >= Min(S) (an toàn)
        obstacle_detected = front_distance <= self.warning_distance and len(front_ranges) >= self.min_obstacle_points

        if front_distance <= self.safety_distance:
            danger_level = 'danger'
        elif front_distance <= self.warning_distance:
            danger_level = 'warning'
        else:
            danger_level = 'safe'

        # === QUYẾT ĐỊNH HƯỚNG NÉ (áp dụng U3 - Lane vacancy rate) ===
        # Chọn hướng né dựa trên khoảng trống bên nào lớn hơn
        if obstacle_detected:
            if left_clearance >= right_clearance:
                avoid_direction = 'left'
            else:
                avoid_direction = 'right'
        else:
            avoid_direction = 'none'

        result = {
            'obstacle_detected': obstacle_detected,
            'front_distance': front_distance,
            'left_clearance': left_clearance,
            'right_clearance': right_clearance,
            'avoid_direction': avoid_direction,
            'danger_level': danger_level,
            'front_points': len(front_ranges),
        }
        self.last_result = result
        return result

    def _default_result(self):
        """Trả về kết quả mặc định khi không có dữ liệu."""
        return {
            'obstacle_detected': False,
            'front_distance': float('inf'),
            'left_clearance': float('inf'),
            'right_clearance': float('inf'),
            'avoid_direction': 'none',
            'danger_level': 'safe',
            'front_points': 0,
        }
