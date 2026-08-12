# -*- coding: utf-8 -*-
"""
IntersectionNavigator: Bộ điều khiển chuyển hướng mượt tại ngã tư (Smart City - Bài 2).
Thực thi các hành động vật lý bẻ lái RẼ TRÁI, RẼ PHẢI, ĐI THẲNG, ĐI LÙI khi nhận lệnh từ AI.
"""

import time
import rospy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import settings

class TurnAction:
    NONE = "NONE"
    STRAIGHT = "STRAIGHT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    BACKWARD = "BACKWARD"

class IntersectionNavigator:
    def __init__(self, blackboard=None):
        self.blackboard = blackboard
        self.is_executing = False
        self.action_start_time = 0.0
        self.current_action = TurnAction.NONE

    def execute_turn(self, action, racer, camera_processor=None, latest_image=None):
        """
        Thực thi hành động rẽ ngã tư dựa trên lệnh action ('LEFT', 'RIGHT', 'STRAIGHT', 'BACKWARD').
        """
        if not self.is_executing:
            self.is_executing = True
            self.action_start_time = time.time()
            self.current_action = action
            rospy.loginfo(f"[INTERSECTION] >>> KHỞI CHẠY HÀNH ĐỘNG NGÃ TƯ: {action}")

        elapsed = time.time() - self.action_start_time

        def drive(steer_val, speed_val):
            if hasattr(racer, 'steer'):
                racer.steer(steer_val, speed_val)
            elif hasattr(racer, 'set_steering'):
                racer.set_steering(steer_val)
                if hasattr(racer, 'set_throttle'):
                    racer.set_throttle(speed_val)

        # 1. ĐI THẲNG
        if self.current_action == TurnAction.STRAIGHT:
            if elapsed < getattr(settings, 'GO_STRAIGHT_TIME', 1.2):
                drive(0.0, getattr(settings, 'TURN_SPEED_STRAIGHT', 0.20))
                return False
            else:
                rospy.loginfo("[INTERSECTION] HOÀN THÀNH ĐI THẲNG NGÃ TƯ")
                self.is_executing = False
                self.current_action = TurnAction.NONE
                return True

        # 2. RẼ TRÁI
        elif self.current_action == TurnAction.LEFT:
            turn_time = getattr(settings, 'TURN_LEFT_TIME', 1.8)
            steer_val = getattr(settings, 'TURN_LEFT_STEER', -1.00)
            speed_val = getattr(settings, 'TURN_LEFT_SPEED', 0.15)

            if elapsed < turn_time:
                drive(steer_val, speed_val)
                return False
            else:
                rospy.loginfo("[INTERSECTION] HOÀN THÀNH RẼ TRÁI NGÃ TƯ")
                self.is_executing = False
                self.current_action = TurnAction.NONE
                return True

        # 3. RẼ PHẢI
        elif self.current_action == TurnAction.RIGHT:
            turn_time = getattr(settings, 'TURN_RIGHT_TIME', 1.7)
            steer_val = getattr(settings, 'TURN_RIGHT_STEER', 1.00)
            speed_val = getattr(settings, 'TURN_RIGHT_SPEED', 0.15)

            if elapsed < turn_time:
                drive(steer_val, speed_val)
                return False
            else:
                rospy.loginfo("[INTERSECTION] HOÀN THÀNH RẼ PHẢI NGÃ TƯ")
                self.is_executing = False
                self.current_action = TurnAction.NONE
                return True

        # 4. ĐI LÙI
        elif self.current_action == TurnAction.BACKWARD:
            turn_time = getattr(settings, 'GO_BACKWARD_TIME', 1.2)
            steer_val = getattr(settings, 'TURN_BACKWARD_STEER', 0.00)
            speed_val = getattr(settings, 'TURN_BACKWARD_SPEED', -0.18)

            if elapsed < turn_time:
                drive(steer_val, speed_val)
                return False
            else:
                rospy.loginfo("[INTERSECTION] HOÀN THÀNH ĐI LÙI")
                self.is_executing = False
                self.current_action = TurnAction.NONE
                return True

        self.is_executing = False
        return True
