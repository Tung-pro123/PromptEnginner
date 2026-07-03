#!/usr/bin/env python3
"""
Module Phát hiện & Né tránh Vật cản (Obstacle Detector) cho Speed Track.

Thiết kế cho xe JetRacer (Ackermann steering) trên sa bàn Speed Track:
- Quét LiDAR 3 sector: FRONT, LEFT-SIDE, RIGHT-SIDE
- FSM 3 bước: NORMAL → DODGING → REENTERING  
- S-Curve ramp offset mượt mà khi chuyển làn ảo
- Quét sườn trái để xác nhận đã vượt qua vật cản

Áp dụng từ bài báo s43684-021-00015-x:
- Mô hình khoảng cách an toàn tối thiểu Min(S) = V_A * t_d + D_0
- Grid-based phân vùng sector scanning
"""

import numpy as np
import math
from enum import Enum


class AvoidState(Enum):
    """Trạng thái máy FSM né vật cản."""
    NORMAL = 0        # Bám lane bình thường, offset = 0
    DODGING = 1       # Đang lách né (dịch offset sang phải)
    REENTERING = 2    # Đang quay trở lại làn chính


class ObstacleDetector:
    """Phát hiện vật cản + FSM né tránh với S-Curve ramp cho Ackermann steering.
    
    Kết hợp:
    - Sector scanning (FRONT ±15°, SIDE 70°-110°) từ bản main
    - S-Curve ramp offset mượt mà (4px/frame)
    - Quét sườn trái để xác nhận đã vượt hộp cản
    - Tự chọn hướng né trái/phải dựa trên khoảng trống
    """

    def __init__(self,
                 trigger_distance=0.70,
                 side_clear_distance=0.45,
                 dodge_offset_px=55,
                 ramp_step_px=4,
                 front_scan_half_angle=15.0,
                 side_scan_start=70.0,
                 side_scan_end=110.0,
                 lidar_angle_offset=180.0):
        """
        Args:
            trigger_distance: Khoảng cách kích hoạt né (m). 
                             Min(S) = V_A * t_d + D_0 ≈ 0.70m
            side_clear_distance: Khoảng cách sườn trái an toàn trước khi 
                                nhập lại làn (m)
            dodge_offset_px: Số pixel dịch tâm bám khi né (dương = sang phải)
            ramp_step_px: Tốc độ dịch S-Curve mỗi frame (px/frame)
            front_scan_half_angle: Nửa góc quét phía trước (độ)
            side_scan_start: Góc bắt đầu quét sườn (độ)
            side_scan_end: Góc kết thúc quét sườn (độ)
            lidar_angle_offset: Góc bù lắp đặt LiDAR (180° nếu ngược)
        """
        # Tham số LiDAR
        self.trigger_distance = trigger_distance
        self.side_clear_distance = side_clear_distance
        self.front_scan_half_angle = front_scan_half_angle
        self.side_scan_start = side_scan_start
        self.side_scan_end = side_scan_end
        self.lidar_angle_offset = lidar_angle_offset

        # Tham số dịch làn ảo
        self.dodge_offset_px = dodge_offset_px
        self.ramp_step_px = ramp_step_px

        # Trạng thái FSM
        self.state = AvoidState.NORMAL
        self.target_offset_px = 0.0
        self.current_offset_px = 0.0
        self.avoid_direction = 'right'  # Mặc định né sang phải

    def _normalize_angle(self, angle_deg):
        """Chuẩn hóa góc LiDAR về [-180, 180] có bù offset lắp đặt."""
        angle_deg = angle_deg + self.lidar_angle_offset
        angle_deg = (angle_deg + 180) % 360 - 180
        return angle_deg

    def _scan_sector(self, scan_msg, angle_min_deg, angle_max_deg):
        """Quét các tia LiDAR trong một sector góc cho trước.
        
        Args:
            scan_msg: ROS LaserScan message
            angle_min_deg: Góc bắt đầu sector (độ, đã chuẩn hóa)
            angle_max_deg: Góc kết thúc sector (độ, đã chuẩn hóa)
            
        Returns:
            list[float]: Danh sách khoảng cách hợp lệ trong sector
        """
        if scan_msg is None:
            return []

        distances = []
        for i, dist in enumerate(scan_msg.ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            angle_deg = self._normalize_angle(math.degrees(angle))

            if angle_min_deg <= angle_deg <= angle_max_deg:
                if scan_msg.range_min < dist < scan_msg.range_max:
                    distances.append(dist)

        return distances

    def get_front_distance(self, scan_msg):
        """Đo khoảng cách vật cản trước mặt bằng LiDAR.
        
        Quét vùng [-front_scan_half_angle, +front_scan_half_angle] (mặc định ±15°).
        
        Returns:
            float: Khoảng cách tối thiểu (m), inf nếu không có vật cản
        """
        distances = self._scan_sector(
            scan_msg,
            -self.front_scan_half_angle,
            self.front_scan_half_angle
        )
        return min(distances) if distances else float('inf')

    def is_side_clear(self, scan_msg, side='left'):
        """Kiểm tra sườn bên xe đã thoát khỏi vật cản chưa.
        
        Quét sườn trái (70°-110°) hoặc sườn phải (-110° đến -70°).
        Nếu khoảng cách tối thiểu > side_clear_distance → đã vượt qua hộp cản.
        
        Args:
            scan_msg: ROS LaserScan message
            side: 'left' hoặc 'right'
            
        Returns:
            bool: True nếu sườn đã clear
        """
        if side == 'left':
            distances = self._scan_sector(
                scan_msg, self.side_scan_start, self.side_scan_end
            )
        else:
            distances = self._scan_sector(
                scan_msg, -self.side_scan_end, -self.side_scan_start
            )

        if distances:
            return min(distances) > self.side_clear_distance
        return True  # Không có dữ liệu → coi như clear

    def _choose_avoid_direction(self, scan_msg):
        """Chọn hướng né dựa trên khoảng trống bên nào lớn hơn.
        
        Quét sector trái (30°-70°) và sector phải (-70° đến -30°) để so sánh.
        
        Returns:
            str: 'left' hoặc 'right'
        """
        left_distances = self._scan_sector(scan_msg, 30.0, 70.0)
        right_distances = self._scan_sector(scan_msg, -70.0, -30.0)

        left_clearance = min(left_distances) if left_distances else float('inf')
        right_clearance = min(right_distances) if right_distances else float('inf')

        # Trên sa bàn Speed Track, vật cản thường đặt ở giữa lane
        # → mặc định né sang PHẢI (theo quy tắc giao thông)
        # Chỉ né trái nếu bên phải bị chặn rõ ràng
        if right_clearance < 0.30 and left_clearance > right_clearance:
            return 'left'
        return 'right'

    def update(self, scan_msg):
        """Cập nhật FSM né vật cản và tính offset S-Curve.
        
        Gọi hàm này mỗi frame trong vòng lặp chính.
        
        Args:
            scan_msg: ROS LaserScan message (hoặc None)
            
        Returns:
            dict: {
                'state': AvoidState,
                'offset_px': float (offset pixel hiện tại để cộng vào tâm bám),
                'front_distance': float (khoảng cách vật cản phía trước),
                'avoid_direction': str ('left'/'right'/'none'),
            }
        """
        front_dist = self.get_front_distance(scan_msg)

        # === STATE 1: BÁM LÀN BÌNH THƯỜNG ===
        if self.state == AvoidState.NORMAL:
            self.target_offset_px = 0.0

            # Kích hoạt né khi vật cản quá gần
            if front_dist < self.trigger_distance:
                self.avoid_direction = self._choose_avoid_direction(scan_msg)
                # Dịch offset: phải = dương, trái = âm
                if self.avoid_direction == 'right':
                    self.target_offset_px = self.dodge_offset_px
                else:
                    self.target_offset_px = -self.dodge_offset_px
                self.state = AvoidState.DODGING

        # === STATE 2: ĐANG LÁCH NÉ VẬT CẢN ===
        elif self.state == AvoidState.DODGING:
            # Giữ offset né
            if self.avoid_direction == 'right':
                self.target_offset_px = self.dodge_offset_px
            else:
                self.target_offset_px = -self.dodge_offset_px

            # Kiểm tra sườn phía ngược hướng né đã clear chưa
            # Nếu né sang phải → check sườn trái (vật cản đã ở phía sau bên trái)
            check_side = 'left' if self.avoid_direction == 'right' else 'right'
            if self.is_side_clear(scan_msg, side=check_side):
                self.state = AvoidState.REENTERING
                self.target_offset_px = 0.0

        # === STATE 3: NHẬP LẠI LÀN CŨ ===
        elif self.state == AvoidState.REENTERING:
            self.target_offset_px = 0.0

            # Khi offset đã gần bằng 0 → về bình thường
            if abs(self.current_offset_px) < 1.0:
                self.state = AvoidState.NORMAL

        # === S-CURVE RAMP (dịch dần, không giật) ===
        diff = self.target_offset_px - self.current_offset_px
        if abs(diff) > 0.1:
            step = np.sign(diff) * self.ramp_step_px
            if abs(step) > abs(diff):
                self.current_offset_px = self.target_offset_px
            else:
                self.current_offset_px += step
        else:
            self.current_offset_px = self.target_offset_px

        return {
            'state': self.state,
            'offset_px': self.current_offset_px,
            'front_distance': front_dist,
            'avoid_direction': self.avoid_direction if self.state != AvoidState.NORMAL else 'none',
        }

    def reset(self):
        """Reset FSM về trạng thái ban đầu."""
        self.state = AvoidState.NORMAL
        self.target_offset_px = 0.0
        self.current_offset_px = 0.0
        self.avoid_direction = 'right'
