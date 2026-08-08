# -*- coding: utf-8 -*-
"""
FSMManager: Máy trạng thái né vật cản 3 bước (NORMAL, DODGING, REENTERING)
tích hợp Safety Steering Override và Trả làn 2 giai đoạn (Two-Stage Re-entering).
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class RobotState:
    STATE_NORMAL = 1       # Bám làn bình thường
    STATE_DODGING = 2      # Đang lách né vật cản
    STATE_REENTERING = 3   # Đang lượn quay trở lại làn cũ

class FSMManager:
    def __init__(self, blackboard=None):
        self.blackboard = blackboard
        self.state = RobotState.STATE_NORMAL
        self.dodge_direction = 0.0      # 1.0 (Right), -1.0 (Left)
        self.target_offset_px = 0.0
        self.current_offset_px = 0.0
        
        self.dodge_start_time = 0
        self.reenter_start_time = 0
        self.clear_side_count = 0

    def update(self, front_dist, closest_angle, lidar_processor, scan_msg):
        """Cập nhật chuyển giao trạng thái FSM."""
        now = time.time()

        # =====================================================================
        # 1. TRẠNG THÁI 1: STATE_NORMAL -> STATE_DODGING
        # =====================================================================
        if self.state == RobotState.STATE_NORMAL:
            self.target_offset_px = 0.0
            if front_dist < settings.TRIGGER_DIST:
                self.state = RobotState.STATE_DODGING
                self.dodge_start_time = now
                self.clear_side_count = 0
                
                # LiDAR Mirrored Coordinate Swap:
                # góc dương >= 0.0 (Vật cản bên phải) -> Lách trái (-1.0)
                # góc âm < 0.0 (Vật cản bên trái) -> Lách phải (1.0)
                if closest_angle is not None and closest_angle >= 0.0:
                    self.dodge_direction = -1.0
                    self.target_offset_px = -settings.DODGE_OFFSET_PX
                else:
                    self.dodge_direction = 1.0
                    self.target_offset_px = settings.DODGE_OFFSET_PX
                    
                print(f"[FSM] KÍCH HOẠT NÉ VẬT CẢN: dist={front_dist:.2f}m, angle={closest_angle}, dir={'TRAI' if self.dodge_direction == -1.0 else 'PHAI'}")

        # =====================================================================
        # 2. TRẠNG THÁI 2: STATE_DODGING -> STATE_REENTERING
        # =====================================================================
        elif self.state == RobotState.STATE_DODGING:
            # Quét kiểm tra thoát vật cản ở hông xe
            if self.dodge_direction == 1.0: # Đã lách sang phải -> Quét hông trái
                _, side_dist = lidar_processor.get_closest_obstacle_angle_in_range(scan_msg, -150.0, 30.0, max_dist=settings.SIDE_CLEAR_DIST)
            else: # Đã lách sang trái -> Quét hông phải
                _, side_dist = lidar_processor.get_closest_obstacle_angle_in_range(scan_msg, -30.0, 150.0, max_dist=settings.SIDE_CLEAR_DIST)
                
            if side_dist >= settings.SIDE_CLEAR_DIST:
                self.clear_side_count += 1
            else:
                self.clear_side_count = 0

            is_clear = (self.clear_side_count >= settings.CLEAR_FRAMES_REQUIRED)
            is_timeout = (now - self.dodge_start_time > settings.WATCHDOG_TIMEOUT)
            
            if is_clear or is_timeout:
                self.state = RobotState.STATE_REENTERING
                self.reenter_start_time = now
                self.target_offset_px = 0.0
                print(f"[FSM] ĐÃ THOÁT VẬT CẢN -> BẮT ĐẦU TRẢ LÀN 2 GIAI ĐOẠN (State 3)")

        # =====================================================================
        # 3. TRẠNG THÁI 3: STATE_REENTERING -> STATE_NORMAL
        # =====================================================================
        elif self.state == RobotState.STATE_REENTERING:
            reenter_duration = now - self.reenter_start_time
            if abs(self.current_offset_px) < 1.0 and reenter_duration >= settings.MIN_REENTERING_DURATION:
                self.state = RobotState.STATE_NORMAL
                self.dodge_direction = 0.0
                print(f"[FSM] ĐÃ TRẢ LÀN HOÀN TẤT -> VỀ NORMAL")

        # Cập nhật Ramping Offset mượt mà
        step_speed = settings.RAMP_STEP_DODGE_PX if self.state == RobotState.STATE_DODGING else settings.RAMP_STEP_RETURN_PX
        if self.current_offset_px < self.target_offset_px:
            self.current_offset_px = min(self.target_offset_px, self.current_offset_px + step_speed)
        elif self.current_offset_px > self.target_offset_px:
            self.current_offset_px = max(self.target_offset_px, self.current_offset_px - step_speed)

        if self.blackboard:
            self.blackboard.set('state', self.state)
            self.blackboard.set('dodge_direction', self.dodge_direction)
            self.blackboard.set('current_offset_px', self.current_offset_px)

        return self.state, self.dodge_direction, self.current_offset_px

    def get_state_name(self):
        if self.state == RobotState.STATE_NORMAL: return "NORMAL"
        if self.state == RobotState.STATE_DODGING: return "DODGING"
        if self.state == RobotState.STATE_REENTERING: return "REENTERING"
        return "UNKNOWN"
