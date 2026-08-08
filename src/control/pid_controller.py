# -*- coding: utf-8 -*-
"""
PIDController: Bộ điều khiển P-Controller bám làn với Safety Steering Override & Two-Stage Re-entering.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class PIDController:
    def __init__(self, blackboard=None):
        self.blackboard = blackboard

    def compute_steering(self, C_near, fsm_state, dodge_direction, current_offset_px, reenter_duration=0.0):
        """
        Tính toán góc bẻ lái [-1.0, 1.0] dựa trên trung điểm vạch C_near và trạng thái FSM.
        """
        # =====================================================================
        # GIAI ĐOẠN 1 CỦA REENTERING: ÉP LÁI MỞ GÓC XOAY XE (Open-loop Return)
        # =====================================================================
        if fsm_state == 3 and reenter_duration < settings.OPEN_LOOP_RETURN_TIME:
            # Ép lái góc mở lớn 0.50 ngược hướng né
            steering_angle = -settings.OPEN_LOOP_STEER_ANGLE * dodge_direction
            print(f"[STAGE 1 RETURN] Open-loop turn: steer={steering_angle:.2f}")
            return max(settings.MIN_STEERING, min(settings.MAX_STEERING, steering_angle))

        # =====================================================================
        # BÁM LÀN QUA CAMERA (Closed-Loop Vision Tracking)
        # =====================================================================
        error = (C_near + current_offset_px) - settings.IMAGE_CENTER_X
        steering_angle = error * settings.KP

        # Safety Steering Override trong trạng thái DODGING:
        # Đảm bảo xe luôn vật lý lách ra xa vật cản ngay cả khi camera bị lừa bởi nhiễu hộp
        if fsm_state == 2:
            if dodge_direction == 1.0: # Né sang phải
                steering_angle = max(settings.MIN_DODGE_STEERING, steering_angle)
            elif dodge_direction == -1.0: # Né sang trái
                steering_angle = min(-settings.MIN_DODGE_STEERING, steering_angle)

        return max(settings.MIN_STEERING, min(settings.MAX_STEERING, steering_angle))
