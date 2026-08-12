# -*- coding: utf-8 -*-
"""
AutoTurnController: Bộ điều khiển ngã tư TỰ ĐỘNG CHUYỂN HƯỚNG DỰA TRÊN THỊ GIÁC (Pure Vision).
Tự động phát hiện Dải Vạch Trắng Ngang (Zebra Stop Line) + Nhô đầu xe 35cm + Khóa làn mới.
"""

import time
import cv2
import numpy as np
import rospy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import settings

class AutoTurnState:
    LANE_KEEPING = "LANE_KEEPING"
    NUDGE_FORWARD = "NUDGE_FORWARD"
    TURNING = "TURNING"
    LOCKING_LANE = "LOCKING_LANE"

class AutoTurnController:
    def __init__(self, blackboard=None):
        self.blackboard = blackboard
        self.state = AutoTurnState.LANE_KEEPING
        self.state_start_time = 0.0
        self.target_action = "RIGHT"

    def detect_stop_line(self, frame):
        if frame is None:
            return False
            
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.75):int(h * 0.95), :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        white_pixels_per_row = np.sum(thresh == 255, axis=1)
        has_horizontal_line = np.any(white_pixels_per_row > (w * 0.40))
        
        return has_horizontal_line

    def detect_blue_sign(self, frame):
        if frame is None:
            return 'NONE'
            
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.20):int(h * 0.60), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([100, 100, 80])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 150:
                return 'RIGHT'
        return 'NONE'

    def update(self, frame, racer):
        now = time.time()
        
        def drive(steer_val, speed_val):
            if hasattr(racer, 'steer'):
                racer.steer(steer_val, speed_val)
            elif hasattr(racer, 'set_steering'):
                racer.set_steering(steer_val)
                if hasattr(racer, 'set_throttle'):
                    racer.set_throttle(speed_val)

        if self.state == AutoTurnState.LANE_KEEPING:
            sign = self.detect_blue_sign(frame)
            if sign != 'NONE':
                self.target_action = sign

            if self.detect_stop_line(frame):
                rospy.loginfo(f"[AUTO TURN] >>> PHÁT HIỆN VẠCH DỪNG NGÃ TƯ! Chuyển sang NUDGE FORWARD")
                self.state = AutoTurnState.NUDGE_FORWARD
                self.state_start_time = now

        elif self.state == AutoTurnState.NUDGE_FORWARD:
            drive(0.0, 0.18)
            if now - self.state_start_time >= 0.45:
                rospy.loginfo(f"[AUTO TURN] >>> BẮT ĐẦU BẺ LÁI ÔM CUA: {self.target_action}")
                self.state = AutoTurnState.TURNING
                self.state_start_time = now

        elif self.state == AutoTurnState.TURNING:
            steer = 1.00 if self.target_action == 'RIGHT' else -1.00
            drive(steer, 0.15)
            
            if now - self.state_start_time >= 1.50:
                rospy.loginfo(f"[AUTO TURN] >>> KHÓA LÀN MỚI -> TRẢ LÁI PID THẲNG")
                self.state = AutoTurnState.LOCKING_LANE
                self.state_start_time = now

        elif self.state == AutoTurnState.LOCKING_LANE:
            drive(0.0, 0.18)
            if now - self.state_start_time >= 0.3:
                self.state = AutoTurnState.LANE_KEEPING
                rospy.loginfo(f"[AUTO TURN] HOÀN THÀNH TỰ ĐỘNG RẼ NGÃ TƯ!")

        return self.state, (self.state != AutoTurnState.LANE_KEEPING)
