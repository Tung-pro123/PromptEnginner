#!/usr/bin/env python3

import sys
import os
import rospy
import cv2
import numpy as np
import time
from enum import Enum

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from robot.control.racer_controller import RacerController
from robot.control.pid_controller_v1 import PIDController
from sensor_msgs.msg import LaserScan, Image
from src.core.planning.map_navigator import MapNavigator

from src.smart_city.config import SmartCityConfig
from src.smart_city.perception.seg_dual_lane_detector import SegDualLaneDetector
from src.smart_city.perception.cnn_sign_detector import CNNSignDetector
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


class JetBotControllerTask2:
    def __init__(self):
        rospy.loginfo("Khởi tạo Smart City Controller Task 2 (Seg Line + CNN Signs)...")
        self.cfg = SmartCityConfig()

        self.initialize_hardware()

        # PID Controller
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

        # Modules Nhận diện Kiến trúc mới (Bài toán 2)
        self.lane_detector = SegDualLaneDetector(self.cfg, model_path="models/yolo_seg_lane.onnx")
        self.sign_detector = CNNSignDetector(model_path="models/sign_recognizer.onnx", threshold=0.9)
        self.crosswalk_detector = CrosswalkDetector(self.cfg)

        # ROS
        self.latest_image = None
        rospy.Subscriber(self.cfg.camera_topic, Image, self.camera_callback)

        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 1
        
        self.LABEL_TO_DIRECTION_ENUM = {'N': Direction.NORTH, 'E': Direction.EAST, 'S': Direction.SOUTH, 'W': Direction.WEST}

        # State
        self.current_state = None
        self.state_change_time = rospy.get_time()
        self._set_state(RobotState.WAITING_FOR_LINE, initial=True)
        
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

    def initialize_hardware(self):
        try:
            self.robot = RacerController()
        except Exception as e:
            from unittest.mock import Mock
            self.robot = Mock()
            rospy.logwarn(f"Mock Robot: {e}")

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

                # 1. Phát hiện giao lộ (Crosswalk)
                crosswalk_trigger = self.crosswalk_detector.detect(frame)
                time_since_leave = rospy.get_time() - self.state_change_time
                
                if crosswalk_trigger and time_since_leave > self.cfg.crosswalk_cooldown_sec:
                    rospy.loginfo("GIAO LỘ (Crosswalk)! Đang tiến vào...")
                    self._set_state(RobotState.APPROACHING_INTERSECTION)
                    continue

                # 2. Bám line bằng Segmentation
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
                        # Đi vào xử lý biển báo và đèn giao thông
                        self._set_state(RobotState.HANDLING_EVENT)

            elif self.current_state == RobotState.HANDLING_EVENT:
                self.robot.stop()
                
                # Quét biển báo bằng CNN (ONNX/TensorRT)
                signs_detected = self.sign_detector.detect(self.latest_image)
                detected_classes = [s["class"] for s in signs_detected]
                
                # Ưu tiên 1: Đèn đỏ thì đợi
                if 'traffic-light-red' in detected_classes:
                    self._set_state(RobotState.WAITING_RED_LIGHT)
                    continue
                
                # Ưu tiên 2: Xử lý rẽ / cấm
                self.target_turn_action = self.process_signs_and_plan(detected_classes)
                
                if self.target_turn_action is None:
                    self._set_state(RobotState.DEAD_END)
                else:
                    self._set_state(RobotState.TURNING)
                    
            elif self.current_state == RobotState.WAITING_RED_LIGHT:
                self.robot.stop()
                
                # Liên tục kiểm tra xem có đèn xanh chưa
                signs_detected = self.sign_detector.detect(self.latest_image)
                detected_classes = [s["class"] for s in signs_detected]
                
                if 'traffic-light_green' in detected_classes or 'traffic-light-red' not in detected_classes:
                    # Đã hết đèn đỏ, quay lại xử lý Event để quyết định rẽ
                    self._set_state(RobotState.HANDLING_EVENT)

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

            rate.sleep()
            
        self.cleanup()

    def process_signs_and_plan(self, detected_classes):
        """Xử lý rẽ dựa trên kết quả CNN."""
        current_direction = self.DIRECTIONS[self.current_direction_index]
        
        while True:
            planned_label = self.navigator.get_next_direction_label(self.current_node_id, self.planned_path)
            if not planned_label:
                return None
            
            planned_action = self.map_absolute_to_relative(planned_label, current_direction)
            
            # Ưu tiên biển chỉ dẫn cứng
            intended_action = None
            if 'left' in detected_classes: intended_action = 'left'
            elif 'right' in detected_classes: intended_action = 'right'
            elif 'straight' in detected_classes: intended_action = 'straight'
            else: intended_action = planned_action
            
            # Cấm rẽ / Cấm đi thẳng
            is_prohibited = ('forbidden' in detected_classes) # Nếu muốn tuân thủ chặt hơn, có thể parse biển cấm cụ thể
            
            if is_prohibited:
                banned_edge = (self.current_node_id, self.planned_path[self.planned_path.index(self.current_node_id)+1])
                if banned_edge not in self.banned_edges:
                    self.banned_edges.append(banned_edge)
                
                # Tính lại đường đi
                if hasattr(self.navigator, 'find_shortest_path_through_loads'):
                    new_path = self.navigator.find_shortest_path_through_loads(self.current_node_id, self.navigator.end_node, self.banned_edges)
                else:
                    new_path = self.navigator.find_path(self.current_node_id, self.navigator.end_node, self.banned_edges)
                
                if new_path:
                    self.planned_path = new_path
                    continue # Thử lại với đường mới
                return None
            
            # Tính toán next_node
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
        self.robot.turn_angle(degrees)
        if update_main_direction and degrees % 90 == 0 and degrees != 0:
            num_turns = round(degrees / 90)
            self.current_direction_index = (self.current_direction_index + num_turns + 4) % 4
        time.sleep(0.5)

    def cleanup(self):
        self.robot.stop()


if __name__ == '__main__':
    rospy.init_node('smart_city_task2_node', anonymous=True)
    try:
        ctrl = JetBotControllerTask2()
        ctrl.run()
    except rospy.ROSInterruptException:
        pass
