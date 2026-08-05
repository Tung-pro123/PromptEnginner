#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from jetracer_smartcity.msg import Detection, DetectionArray, IntersectionDecision
from state_machine.route_planner import RoutePlanner
from utils.config_loader import load_yaml
from utils.run_logger import RunLogger
import time

class IntersectionStateMachine:
    def __init__(self):
        rospy.init_node('intersection_state_machine', anonymous=True)
        
        # Load map configuration
        self.map_config_path = rospy.get_param('~map_config_path', 'd:/AI_Project/racing_promax/catkin_ws/src/jetracer_smartcity/config/intersection_map.yaml')
        self.route_planner = RoutePlanner(self.map_config_path)
        
        # State definitions
        # CRUISING: follow lane
        # APPROACH_NODE: detected sign/light, slowing down
        # WAIT_SIGNAL: red light, waiting
        # DECIDE_DIRECTION: pathfinding and sign compliance
        # EXECUTE_TURN: executing turn maneuver
        self.state = "CRUISING"
        
        # Positions
        self.current_node = self.route_planner.start_node
        self.prev_node = None
        self.target_path = self.route_planner.find_path(self.current_node, self.route_planner.finish_node)
        self.next_node = self.target_path[1] if self.target_path and len(self.target_path) > 1 else None
        
        rospy.loginfo("Initial path: %s", str(self.target_path))
        
        # State parameters
        self.min_detection_conf = rospy.get_param('~min_detection_conf', 0.45)
        self.approach_box_size = rospy.get_param('~approach_box_size', 40) # min height in pixels to trigger approach
        
        # Timers and state vars
        self.state_timer = 0.0
        self.turn_duration = rospy.get_param('~turn_duration', 1.8) # seconds to execute open-loop turn
        self.last_decision = "CRUISE"
        
        # Initialize Logger
        self.logger = RunLogger()
        self.fps_avg = 30.0 # Default value
        
        # ROS Publishers
        self.decision_pub = rospy.Publisher('/decision', IntersectionDecision, queue_size=10)
        
        # ROS Subscribers
        self.detections_sub = rospy.Subscriber('/detections', DetectionArray, self.detections_callback)
        self.fps_sub = rospy.Subscriber('/pipeline_fps', String, self.fps_callback)
        
        rospy.loginfo("Intersection State Machine initialized in state CRUISING at node %s", self.current_node)

    def fps_callback(self, msg):
        try:
            self.fps_avg = float(msg.data)
        except ValueError:
            pass

    def detections_callback(self, msg):
        """
        Receives object detections and updates state machine transitions.
        """
        detections = msg.detections
        
        # Process state transitions
        if self.state == "CRUISING":
            # Search for signs or traffic lights indicating an upcoming node
            upcoming_node_detected = False
            for det in detections:
                if det.confidence >= self.min_detection_conf:
                    # Bounding box height check to ensure vehicle is close enough
                    if det.h >= self.approach_box_size:
                        upcoming_node_detected = True
                        break
            
            if upcoming_node_detected:
                self.transition_to("APPROACH_NODE")
                
        elif self.state == "APPROACH_NODE":
            # Handle traffic lights and signs
            traffic_light = None
            for det in detections:
                if "light" in det.label and det.confidence >= self.min_detection_conf:
                    traffic_light = det
                    break
            
            if traffic_light:
                # If red light, wait
                if traffic_light.label == "red_light":
                    self.transition_to("WAIT_SIGNAL")
                else:
                    self.transition_to("DECIDE_DIRECTION")
            else:
                # No light detected, proceed directly to decision
                self.transition_to("DECIDE_DIRECTION")
                
        elif self.state == "WAIT_SIGNAL":
            # Wait until light is green
            green_detected = False
            for det in detections:
                if det.label == "green_light" and det.confidence >= self.min_detection_conf:
                    green_detected = True
                    break
            
            if green_detected:
                self.transition_to("DECIDE_DIRECTION")
                
        elif self.state == "DECIDE_DIRECTION":
            # Read signs to alter path if necessary
            self.handle_signs_and_replan(detections)
            self.make_routing_decision()
            
        elif self.state == "EXECUTE_TURN":
            # Timed transition back to cruising after turn execution
            if time.time() - self.state_timer >= self.turn_duration:
                # Arrived at next node
                self.prev_node = self.current_node
                self.current_node = self.next_node
                
                # Re-plan path from new current node
                self.target_path = self.route_planner.find_path(self.current_node, self.route_planner.finish_node)
                self.next_node = self.target_path[1] if self.target_path and len(self.target_path) > 1 else None
                
                rospy.loginfo("Turn complete. Arrived at node: %s. Next target node: %s", self.current_node, self.next_node)
                self.transition_to("CRUISING")

    def transition_to(self, new_state):
        rospy.loginfo("State transition: %s -> %s", self.state, new_state)
        self.state = new_state
        self.state_timer = time.time()
        
        # Publish quick update if needed
        if new_state == "WAIT_SIGNAL":
            self.publish_decision("STOP")
        elif new_state == "CRUISING":
            self.publish_decision("CRUISE")

    def handle_signs_and_replan(self, detections):
        """
        Processes sign detections and blocks/modifies graph paths dynamically.
        """
        if not self.next_node:
            return
            
        for det in detections:
            if det.confidence < self.min_detection_conf:
                continue
                
            # Prohibitory Signs
            if det.label == "no_left_sign":
                # Find neighbor corresponding to Left and block it
                left_node = self.find_neighbor_by_turn("left")
                if left_node:
                    self.route_planner.block_edge(self.current_node, left_node)
                    self.target_path = self.route_planner.find_path(self.current_node, self.route_planner.finish_node)
                    
            elif det.label == "no_right_sign":
                # Find neighbor corresponding to Right and block it
                right_node = self.find_neighbor_by_turn("right")
                if right_node:
                    self.route_planner.block_edge(self.current_node, right_node)
                    self.target_path = self.route_planner.find_path(self.current_node, self.route_planner.finish_node)
                    
            elif det.label == "no_straight_sign":
                straight_node = self.find_neighbor_by_turn("straight")
                if straight_node:
                    self.route_planner.block_edge(self.current_node, straight_node)
                    self.target_path = self.route_planner.find_path(self.current_node, self.route_planner.finish_node)
                    
            # Mandatory Signs
            elif det.label == "turn_left_sign":
                left_node = self.find_neighbor_by_turn("left")
                if left_node:
                    # Enforce left by blocking all other connections from current node
                    self.enforce_single_direction(left_node)
                    
            elif det.label == "turn_right_sign":
                right_node = self.find_neighbor_by_turn("right")
                if right_node:
                    self.enforce_single_direction(right_node)
                    
            elif det.label == "go_straight_sign":
                straight_node = self.find_neighbor_by_turn("straight")
                if straight_node:
                    self.enforce_single_direction(straight_node)

    def enforce_single_direction(self, target_neighbor):
        """
        Blocks all edges from the current node except the target neighbor to force a turn.
        """
        for conn in self.route_planner.nodes[self.current_node]["connections"]:
            if conn != target_neighbor:
                self.route_planner.block_edge(self.current_node, conn)
        self.target_path = self.route_planner.find_path(self.current_node, self.route_planner.finish_node)

    def find_neighbor_by_turn(self, turn_dir):
        """
        Helper to scan neighbors of the current node and return the one matching the relative turn direction.
        """
        for neighbor in self.route_planner.nodes[self.current_node]["connections"]:
            rel_dir = self.route_planner.get_required_direction(self.prev_node, self.current_node, neighbor)
            if rel_dir == turn_dir:
                return neighbor
        return None

    def make_routing_decision(self):
        """
        Decides the turning instruction at the intersection based on the target path.
        """
        if not self.target_path or len(self.target_path) < 2:
            rospy.logwarn("No path found to target!")
            self.publish_decision("STOP")
            return
            
        self.next_node = self.target_path[1]
        
        # Calculate relative turn direction using route planner geometry
        direction = self.route_planner.get_required_direction(self.prev_node, self.current_node, self.next_node)
        
        rospy.loginfo("Routing decision: Go %s from %s to %s", direction.upper(), self.current_node, self.next_node)
        
        decision_cmd = "CRUISE"
        if direction == "left":
            decision_cmd = "TURN_LEFT"
        elif direction == "right":
            decision_cmd = "TURN_RIGHT"
        elif direction == "straight":
            decision_cmd = "GO_STRAIGHT"
            
        self.publish_decision(decision_cmd)
        self.transition_to("EXECUTE_TURN")

    def publish_decision(self, decision_str):
        self.last_decision = decision_str
        
        msg = IntersectionDecision()
        msg.header.stamp = rospy.Time.now()
        msg.decision = decision_str
        msg.node_id = self.current_node
        msg.latency_ms = 0.0 # Latency is calculated at the detector node
        
        self.decision_pub.publish(msg)
        
        # Log event to logger
        self.logger.log_event(
            fps=self.fps_avg,
            detected_object="N/A",
            confidence=0.0,
            decision=decision_str,
            latency_ms=0.0,
            control_output="N/A",
            event=f"State: {self.state} Node: {self.current_node}"
        )

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = IntersectionStateMachine()
        node.spin()
    except rospy.ROSInterruptException:
        pass
