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
from src.smart_city.perception.dual_lane_detector import DualLaneDetector
from src.smart_city.perception.crosswalk_detector import CrosswalkDetector

class RobotState(Enum):
    WAITING_FOR_LINE = 0
    DRIVING_STRAIGHT = 1
    APPROACHING_INTERSECTION = 2
    WAITING_RED_LIGHT = 3       # Chờ đèn đỏ
    HANDLING_EVENT = 4          # Xử lý biển báo
    TURNING = 5                 # Đang rẽ
    LEAVING_INTERSECTION = 6
    REACQUIRING_LINE = 7
    DEAD_END = 8
    GOAL_REACHED = 9

class Direction(Enum):
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3


class JetBotControllerV2:
    def __init__(self):
        rospy.loginfo("Khởi tạo Smart City Controller v2 (Dual Lane)...")
        self.cfg = SmartCityConfig()

        self.initialize_hardware()
        self.initialize_yolo()
        self.initialize_mqtt()

        self.video_writer = None
        self.initialize_video_writer()

        # PID Controller (thay vì tự tính)
        self.pid = PIDController(
            kp=self.cfg.pid_kp, ki=self.cfg.pid_ki, kd=self.cfg.pid_kd,
            output_min=-self.cfg.max_correction, output_max=self.cfg.max_correction
        )

        # Map Navigation
        map_path = os.path.join(os.path.dirname(__file__), "..", "..", "core", "utils", self.cfg.map_filename)
        self.navigator = MapNavigator(map_path)
        self.current_node_id = self.navigator.start_node
        self.target_node_id = None
        self.planned_path = None
        self.banned_edges = []
        self.plan_initial_route()

        # Perception Modules
        self.lane_detector = DualLaneDetector(self.cfg)
        self.crosswalk_detector = CrosswalkDetector(self.cfg)
        self.lidar_detector = SimpleOppositeDetector()

        # ROS
        self.latest_scan = None
        self.latest_image = None
        rospy.Subscriber(self.cfg.lidar_topic, LaserScan, self.lidar_detector.callback)
        rospy.Subscriber(self.cfg.camera_topic, Image, self.camera_callback)

        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 1
        self.ANGLE_TO_FACE_SIGN_MAP = {d: a for d, a in zip(self.DIRECTIONS, [45, -45, -135, 135])}
        
        self.LABEL_TO_DIRECTION_ENUM = {'N': Direction.NORTH, 'E': Direction.EAST, 'S': Direction.SOUTH, 'W': Direction.WEST}
        self.PRESCRIPTIVE_SIGNS = {'N', 'E', 'W', 'S'}
        self.PROHIBITIVE_SIGNS = {'NN', 'NE', 'NW', 'NS'}
        self.DATA_ITEMS = {'qr_code', 'math_problem'}

        # State
        self.current_state = None
        self.state_change_time = rospy.get_time()
        self._set_state(RobotState.WAITING_FOR_LINE, initial=True)
        
        # Thêm biến lưu target_action để rẽ
        self.target_turn_action = None

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
            fname = f'smart_city_v2_{int(time.time())}.avi'
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

    def initialize_yolo(self):
        self.yolo_session = None
        self.submit_module = self._load_submit_module()

    def _load_submit_module(self):
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, "submit_sign copy.py")
        if not os.path.exists(path):
            return None
        try:
            spec = importlib.util.spec_from_file_location("submit_sign_copy", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            rospy.logerr(f"Lỗi load submit script: {e}")
            return None

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
            
            if self.current_state == RobotState.WAITING_FOR_LINE:
                self.robot.stop()
                if frame is not None and self.lane_detector.is_line_visible(frame):
                    rospy.loginfo("Đã thấy line. Bắt đầu.")
                    self._set_state(RobotState.DRIVING_STRAIGHT)

            elif self.current_state == RobotState.DRIVING_STRAIGHT:
                if frame is None:
                    continue

                # 1. Phát hiện giao lộ (LiDAR hoặc Crosswalk)
                lidar_trigger = self.lidar_detector.process_detection()
                crosswalk_trigger = self.crosswalk_detector.detect(frame)
                
                # Cooldown cho crosswalk (chỉ tính crosswalk khi cooldown xong)
                time_since_leave = rospy.get_time() - self.state_change_time
                if (lidar_trigger) or (crosswalk_trigger and time_since_leave > self.cfg.crosswalk_cooldown_sec):
                    rospy.loginfo("GIAO LỘ! Đang tiến vào...")
                    self._set_state(RobotState.APPROACHING_INTERSECTION)
                    continue

                # 2. Bám line
                res_exec = self.lane_detector.get_execution_center(frame)
                if res_exec.center_x is not None:
                    error = res_exec.center_x - (self.cfg.image_width / 2)
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
                        # Thay vì rẽ luôn, chuyển sang kiểm tra đèn đỏ/biển báo
                        # Hiện tại chưa có detect_traffic_light, đi thẳng qua HANDLING_EVENT
                        # Nếu có AI Traffic Light: self._set_state(RobotState.WAITING_RED_LIGHT)
                        self._set_state(RobotState.HANDLING_EVENT)

            elif self.current_state == RobotState.HANDLING_EVENT:
                # Dừng lại xử lý biển báo
                self.robot.stop()
                time.sleep(0.5)
                
                # Quét biển
                self.target_turn_action = self.process_signs_and_plan()
                
                if self.target_turn_action is None:
                    self._set_state(RobotState.DEAD_END)
                else:
                    self._set_state(RobotState.TURNING)

            elif self.current_state == RobotState.TURNING:
                if self.target_turn_action == 'straight':
                    pass
                elif self.target_turn_action == 'right':
                    self.turn_robot(90, True)
                elif self.target_turn_action == 'left':
                    self.turn_robot(-90, True)
                
                self._set_state(RobotState.LEAVING_INTERSECTION)

            elif self.current_state == RobotState.LEAVING_INTERSECTION:
                self.robot.forward(self.cfg.base_speed)
                if rospy.get_time() - self.state_change_time > self.cfg.intersection_clearance_duration:
                    self._set_state(RobotState.REACQUIRING_LINE)

            elif self.current_state == RobotState.REACQUIRING_LINE:
                self.robot.forward(self.cfg.recover_speed)
                if frame is not None and self.lane_detector.is_line_visible(frame):
                    self._set_state(RobotState.DRIVING_STRAIGHT)
                elif rospy.get_time() - self.state_change_time > self.cfg.line_reacquire_timeout:
                    self._set_state(RobotState.DEAD_END)

            elif self.current_state == RobotState.DEAD_END or self.current_state == RobotState.GOAL_REACHED:
                self.robot.stop()
                break

            self._record_frame()
            rate.sleep()
            
        self.cleanup()

    def process_signs_and_plan(self):
        """Xử lý biển báo và tính toán rẽ. Return: 'left', 'right', 'straight', None"""
        current_direction = self.DIRECTIONS[self.current_direction_index]
        angle_to_sign = self.ANGLE_TO_FACE_SIGN_MAP.get(current_direction, 0)
        
        self.turn_robot(angle_to_sign, False)
        detections = self.detect_with_yolo(self.latest_image)
        self.turn_robot(-angle_to_sign, False)
        
        prescriptive_cmds = {d['class_name'] for d in detections if d['class_name'] in self.PRESCRIPTIVE_SIGNS}
        prohibitive_cmds = {d['class_name'] for d in detections if d['class_name'] in self.PROHIBITIVE_SIGNS}
        
        # Gửi payload lên server
        for det in detections:
            payload = {
                "text": f"Detected: {det.get('class_name')}, conf={det.get('confidence')}",
                "race": 1,
                "node_id": int(self.current_node_id) if str(self.current_node_id).isdigit() else self.current_node_id,
                "submit_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace('+00:00','Z'),
                "team": self.cfg.team_name if hasattr(self.cfg, 'team_name') else 'Tên đội'
            }
            # Dummy HTTP logic (porting from old code)
            try:
                headers = {'Content-Type': 'application/json'}
                requests.post('http://example.com/submit', json=payload, headers=headers, timeout=2)
            except: pass

        while True:
            planned_label = self.navigator.get_next_direction_label(self.current_node_id, self.planned_path)
            if not planned_label:
                return None
            
            planned_action = self.map_absolute_to_relative(planned_label, current_direction)
            
            intended_action = None
            if 'L' in prescriptive_cmds: intended_action = 'left'
            elif 'R' in prescriptive_cmds: intended_action = 'right'
            elif 'F' in prescriptive_cmds: intended_action = 'straight'
            else: intended_action = planned_action
            
            is_prohibited = (intended_action == 'straight' and 'NF' in prohibitive_cmds) or \
                            (intended_action == 'right' and 'NR' in prohibitive_cmds) or \
                            (intended_action == 'left' and 'NL' in prohibitive_cmds)
            
            if is_prohibited:
                banned_edge = (self.current_node_id, self.planned_path[self.planned_path.index(self.current_node_id)+1])
                if banned_edge not in self.banned_edges:
                    self.banned_edges.append(banned_edge)
                if hasattr(self.navigator, 'find_shortest_path_through_loads'):
                    new_path = self.navigator.find_shortest_path_through_loads(self.current_node_id, self.navigator.end_node, self.banned_edges)
                else:
                    new_path = self.navigator.find_path(self.current_node_id, self.navigator.end_node, self.banned_edges)
                if new_path:
                    self.planned_path = new_path
                    continue
                return None
            
            # Tính toán next_node
            new_robot_direction = current_direction # simplification
            next_node_id = self.navigator.get_neighbor_by_direction(self.current_node_id, planned_label)
            if next_node_id:
                self.target_node_id = next_node_id
            
            return intended_action

    def map_absolute_to_relative(self, target_label, current_dir):
        target_dir = self.LABEL_TO_DIRECTION_ENUM.get(target_label)
        if not target_dir: return None
        diff = (target_dir.value - current_dir.value + 4) % 4 
        if diff == 0: return 'straight'
        elif diff == 1: return 'right'
        elif diff == 3: return 'left'
        return 'turn_around'

    def turn_robot(self, degrees, update_main_direction=True):
        self.robot.turn_angle(degrees, record_callback=self._record_frame)
        if update_main_direction and degrees % 90 == 0 and degrees != 0:
            num_turns = round(degrees / 90)
            self.current_direction_index = (self.current_direction_index + num_turns + 4) % 4
        time.sleep(0.5)

    def detect_with_yolo(self, image):
        if image is None or not self.cfg.rf_api_key: return []
        try:
            _, img_encoded = cv2.imencode('.jpg', image)
            files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
            url = f"https://detect.roboflow.com/{self.cfg.rf_model}/{self.cfg.rf_version}"
            params = {'api_key': self.cfg.rf_api_key}
            resp = requests.post(url, params=params, files=files, timeout=8)
            if resp.status_code == 200:
                return resp.json().get('predictions', [])
        except: pass
        return []

    def _record_frame(self):
        if self.video_writer and self.latest_image is not None:
            frame = self.latest_image.copy()
            res_exec = self.lane_detector.get_execution_center(frame)
            res_look = self.lane_detector.get_lookahead_center(frame)
            crosswalk = self.crosswalk_detector.detect(frame)
            
            frame = self.lane_detector.draw_debug(frame, res_exec, res_look)
            frame = self.crosswalk_detector.draw_debug(frame, crosswalk)
            
            cv2.putText(frame, f"State: {self.current_state.name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.video_writer.write(frame)

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
