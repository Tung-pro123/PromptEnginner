#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import rospy
import cv2
import numpy as np
import time
import json
import math
from enum import Enum
import requests
import importlib.util
from datetime import datetime, timezone

from src.core.control.racer_controller import RacerController
from src.core.control.pid_controller import PIDController
import paho.mqtt.client as mqtt
from sensor_msgs.msg import LaserScan, Image
from src.core.utils.opposite_detector import SimpleOppositeDetector
from src.core.planning.map_navigator import MapNavigator

from src.smart_city.config import SmartCityConfig
from src.smart_city.perception.seg_lane_detector import YoloSegDetector
from src.smart_city.perception.traffic_detector import TrafficDetector, TrafficSign, TrafficLight

class RobotState(Enum):
    WAITING_FOR_LINE = 0
    DRIVING_STRAIGHT = 1
    APPROACHING_INTERSECTION = 2
    WAITING_RED_LIGHT = 3
    HANDLING_EVENT = 4
    TURNING = 5
    LEAVING_INTERSECTION = 6
    REACQUIRING_LINE = 7
    DEAD_END = 8
    GOAL_REACHED = 9

class Direction(Enum):
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3


class JetBotControllerV2:
    def __init__(self):
        rospy.loginfo("Khởi tạo Smart City Controller v2 (YOLO-seg + Traffic Detector)...")
        self.cfg = SmartCityConfig()

        self.initialize_hardware()
        self.initialize_mqtt()
        self.initialize_video_writer()

        # Cấu hình PID Controller
        self.pid = PIDController(
            kp=self.cfg.pid_kp, ki=self.cfg.pid_ki, kd=self.cfg.pid_kd,
            output_min=-self.cfg.max_correction, output_max=self.cfg.max_correction
        )

        # Cấu hình Map Navigation
        map_path = os.path.join(os.path.dirname(__file__), "..", "..", "core", "utils", self.cfg.map_filename)
        self.navigator = MapNavigator(map_path)
        self.current_node_id = self.navigator.start_node
        self.target_node_id = None
        self.planned_path = None
        self.banned_edges = []
        self.plan_initial_route()

        # === PERCEPTION MODULES MỚI ===
        # Tích hợp nhận diện YOLO Segmentation thay cho HSV DualLaneDetector
        # Tìm file ONNX hoặc Engine (Ưu tiên engine nếu có)
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best.engine')
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best.onnx')
            if not os.path.exists(model_path):
                # Fallback pt
                model_path = "yolov8n-seg.pt"
                
        self.seg_detector = YoloSegDetector(model_path, self.cfg)
        self.sign_detector = TrafficDetector(self.cfg.image_width, self.cfg.image_height)
        self.lidar_detector = SimpleOppositeDetector()

        # ROS
        self.latest_scan = None
        self.latest_image = None
        rospy.Subscriber(self.cfg.lidar_topic, LaserScan, self.lidar_detector.callback)
        rospy.Subscriber(self.cfg.camera_topic, Image, self.camera_callback)

        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 1
        
        # State
        self.current_state = None
        self.state_change_time = rospy.get_time()
        self._set_state(RobotState.WAITING_FOR_LINE, initial=True)
        
        self.target_turn_action = None
        self.turn_start_time = 0.0

        rospy.loginfo("Khởi tạo hoàn tất.")

    def plan_initial_route(self): 
        rospy.loginfo(f"Lập kế hoạch từ {self.navigator.start_node} đến {self.navigator.end_node}...")
        if hasattr(self.navigator, 'find_shortest_path_through_loads'):
            self.planned_path = self.navigator.find_shortest_path_through_loads(
                self.navigator.start_node, self.navigator.end_node, self.banned_edges
            )
        else:
            self.planned_path = self.navigator.find_path(
                self.navigator.start_node, self.navigator.end_node, self.banned_edges
            )

        if self.planned_path and len(self.planned_path) > 1:
            self.target_node_id = self.planned_path[1]
            rospy.loginfo(f"Đường đi: {self.planned_path}")
        else:
            rospy.logerr("Lỗi đường đi!")
            self._set_state(RobotState.DEAD_END)

    def initialize_video_writer(self):
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fname = f'smart_city_yolo_seg_{int(time.time())}.avi'
            self.video_writer = cv2.VideoWriter(fname, fourcc, self.cfg.video_fps, 
                                                (self.cfg.image_width, self.cfg.image_height))
            if self.video_writer.isOpened():
                rospy.loginfo(f"Ghi video: {fname}")
            else:
                self.video_writer = None
        except Exception as e:
            rospy.logerr(f"Lỗi VideoWriter: {e}")
            self.video_writer = None

    def initialize_hardware(self):
        try:
            self.robot = RacerController()
        except Exception as e:
            from unittest.mock import Mock
            self.robot = Mock()
            rospy.logwarn(f"Mock Robot: {e}")

    def initialize_mqtt(self):
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = lambda c,u,f,rc: rospy.loginfo(f"MQTT: {rc}")
        try:
            self.mqtt_client.connect(self.cfg.mqtt_broker, self.cfg.mqtt_port, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            rospy.logerr(f"Lỗi MQTT: {e}")

    def _set_state(self, new_state, initial=False):
        if self.current_state != new_state:
            if not initial: rospy.loginfo(f"STATE: {self.current_state.name} -> {new_state.name}")
            self.current_state = new_state
            self.state_change_time = rospy.get_time()
            if new_state == RobotState.DRIVING_STRAIGHT:
                self.pid.reset()

    def camera_callback(self, msg):
        try:
            if 'compressed' in msg.encoding:
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                if 'rgb' in msg.encoding: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.latest_image = cv2.resize(img, (self.cfg.image_width, self.cfg.image_height))
        except Exception as e:
            pass

    def run(self):
        rospy.loginfo("Đợi 3s...")
        time.sleep(3)
        self.lidar_detector.start_scanning()
        rate = rospy.Rate(self.cfg.loop_rate)
        
        while not rospy.is_shutdown():
            frame = self.latest_image
            
            if frame is None:
                rate.sleep()
                continue
                
            # YOLO-seg nhận diện liên tục
            seg_result = self.seg_detector.detect(frame)

            if self.current_state == RobotState.WAITING_FOR_LINE:
                self.robot.stop()
                if seg_result.mode != 'LOST':
                    rospy.loginfo("Đã thấy line qua YOLO. Bắt đầu.")
                    self._set_state(RobotState.DRIVING_STRAIGHT)

            elif self.current_state == RobotState.DRIVING_STRAIGHT:
                # 1. Phát hiện giao lộ
                lidar_trigger = self.lidar_detector.process_detection()
                crosswalk_trigger = seg_result.has_crosswalk
                
                time_since_leave = rospy.get_time() - self.state_change_time
                if (lidar_trigger) or (crosswalk_trigger and time_since_leave > self.cfg.crosswalk_cooldown_sec):
                    rospy.loginfo("GIAO LỘ! Đang tiến vào...")
                    self._set_state(RobotState.APPROACHING_INTERSECTION)
                    continue

                # 2. Bám line bình thường (Center of Left and Right masks)
                if seg_result.center_x is not None:
                    error = seg_result.center_x - (self.cfg.image_width / 2)
                    steer = self.pid.compute(error)
                    speed = self.cfg.curve_speed if abs(error) > self.cfg.curve_error_thresh else self.cfg.base_speed
                    self.robot.steer(steer, speed)
                else:
                    self.robot.stop()
                    rospy.logwarn("Mất line đột ngột!")

            elif self.current_state == RobotState.APPROACHING_INTERSECTION:
                self.robot.forward(self.cfg.intersection_speed)
                if rospy.get_time() - self.state_change_time > self.cfg.intersection_approach_duration:
                    self.robot.stop()
                    self.current_node_id = self.target_node_id
                    
                    if self.current_node_id == self.navigator.end_node:
                        self._set_state(RobotState.GOAL_REACHED)
                    else:
                        self._set_state(RobotState.HANDLING_EVENT)

            elif self.current_state == RobotState.HANDLING_EVENT:
                self.robot.stop()
                time.sleep(0.5) # Đợi xe ổn định
                
                # Sử dụng thuật toán nhận diện biển báo Computer Vision tích hợp mới
                light, sign = self.sign_detector.detect(self.latest_image)
                rospy.loginfo(f"Phát hiện biển báo: {sign}")
                
                if sign == TrafficSign.LEFT:
                    self.target_turn_action = 'left'
                elif sign == TrafficSign.RIGHT:
                    self.target_turn_action = 'right'
                elif sign == TrafficSign.STRAIGHT:
                    self.target_turn_action = 'straight'
                else:
                    # Nếu không thấy biển báo, đi theo thuật toán tìm đường MapNavigator
                    current_direction = self.DIRECTIONS[self.current_direction_index]
                    planned_label = self.navigator.get_next_direction_label(self.current_node_id, self.planned_path)
                    
                    if planned_label:
                        # Convert L, R, F from absolute
                        target_dir = {'N': Direction.NORTH, 'E': Direction.EAST, 'S': Direction.SOUTH, 'W': Direction.WEST}.get(planned_label)
                        diff = (target_dir.value - current_direction.value + 4) % 4 
                        if diff == 0: self.target_turn_action = 'straight'
                        elif diff == 1: self.target_turn_action = 'right'
                        elif diff == 3: self.target_turn_action = 'left'
                        else: self.target_turn_action = 'straight'
                        
                        next_node_id = self.navigator.get_neighbor_by_direction(self.current_node_id, planned_label)
                        if next_node_id:
                            self.target_node_id = next_node_id
                    else:
                        self.target_turn_action = 'straight'

                self.turn_start_time = rospy.get_time()
                self._set_state(RobotState.TURNING)

            elif self.current_state == RobotState.TURNING:
                # Target Lane Switching (Chuyển đổi trọng số bám đường)
                
                if self.target_turn_action == 'straight':
                    self._set_state(RobotState.LEAVING_INTERSECTION)
                    continue
                    
                target_setpoint = None
                
                # Ép xe bám theo một cạnh duy nhất
                if self.target_turn_action == 'right':
                    if seg_result.right_x is not None:
                        target_setpoint = seg_result.right_x - self.cfg.lane_half_width_px
                    elif seg_result.left_x is not None: # Dự phòng nếu chỉ thấy left
                        target_setpoint = seg_result.left_x + self.cfg.lane_half_width_px
                        
                elif self.target_turn_action == 'left':
                    if seg_result.left_x is not None:
                        target_setpoint = seg_result.left_x + self.cfg.lane_half_width_px
                    elif seg_result.right_x is not None:
                        target_setpoint = seg_result.right_x - self.cfg.lane_half_width_px
                
                if target_setpoint is not None:
                    error = target_setpoint - (self.cfg.image_width / 2)
                    steer = self.pid.compute(error)
                    self.robot.steer(steer, self.cfg.curve_speed)
                else:
                    # Mất cả 2 lane, chạy mù chậm để đợi
                    steer_blind = 1.0 if self.target_turn_action == 'right' else -1.0
                    self.robot.steer(steer_blind, self.cfg.curve_speed)

                # Điều kiện thoát trạng thái rẽ:
                # Phải qua ít nhất 1.5s và xe nhìn thấy đủ cả 2 lane của đường thẳng mới
                time_in_turn = rospy.get_time() - self.turn_start_time
                if time_in_turn > 1.5 and seg_result.mode == 'BOTH':
                    rospy.loginfo("Hoàn tất cua! Đã lấy lại được 2 làn đường mới.")
                    # Cập nhật hướng la bàn ảo
                    if self.target_turn_action == 'right':
                        self.current_direction_index = (self.current_direction_index + 1) % 4
                    elif self.target_turn_action == 'left':
                        self.current_direction_index = (self.current_direction_index + 3) % 4
                        
                    self._set_state(RobotState.LEAVING_INTERSECTION)

            elif self.current_state == RobotState.LEAVING_INTERSECTION:
                self.robot.forward(self.cfg.base_speed)
                if rospy.get_time() - self.state_change_time > self.cfg.intersection_clearance_duration:
                    self._set_state(RobotState.REACQUIRING_LINE)

            elif self.current_state == RobotState.REACQUIRING_LINE:
                self.robot.forward(self.cfg.recover_speed)
                if seg_result.mode != 'LOST':
                    self._set_state(RobotState.DRIVING_STRAIGHT)
                elif rospy.get_time() - self.state_change_time > self.cfg.line_reacquire_timeout:
                    self._set_state(RobotState.DEAD_END)

            elif self.current_state == RobotState.DEAD_END or self.current_state == RobotState.GOAL_REACHED:
                self.robot.stop()
                break

            self._record_frame(frame, seg_result)
            rate.sleep()
            
        self.cleanup()


    def _record_frame(self, frame, seg_result):
        if self.video_writer and frame is not None:
            # Dùng luôn hàm draw_debug của YOLO Seg
            frame_debug = self.seg_detector.draw_debug(frame.copy(), seg_result)
            cv2.putText(frame_debug, f"State: {self.current_state.name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.video_writer.write(frame_debug)

    def cleanup(self):
        self.robot.stop()
        if self.video_writer: self.video_writer.release()
        self.lidar_detector.stop_scanning()
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()


if __name__ == '__main__':
    rospy.init_node('smart_city_v2_node', anonymous=True)
    try:
        ctrl = JetBotControllerV2()
        ctrl.run()
    except rospy.ROSInterruptException:
        pass
