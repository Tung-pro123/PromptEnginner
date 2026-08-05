#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from jetracer_smartcity.msg import IntersectionDecision
import time

class ControlNode:
    def __init__(self):
        rospy.init_node('control_node', anonymous=True)
        
        # Load parameters
        self.cruise_throttle = rospy.get_param('~cruise_throttle', 0.15)
        self.turn_throttle = rospy.get_param('~turn_throttle', 0.11)
        self.max_steer_left = rospy.get_param('~max_steer_left', -0.8)
        self.max_steer_right = rospy.get_param('~max_steer_right', 0.8)
        self.straight_duration = rospy.get_param('~straight_duration', 1.0) # crossing intersection time
        
        # State variables
        self.current_decision = "CRUISE"
        self.lane_steering = 0.0
        self.decision_time = 0.0
        
        # ROS Publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # ROS Subscribers
        self.steer_sub = rospy.Subscriber('/steering_angle', Float32, self.steering_callback)
        self.decision_sub = rospy.Subscriber('/decision', IntersectionDecision, self.decision_callback)
        
        # Control Loop Timer (30Hz)
        self.timer = rospy.Timer(rospy.Duration(0.033), self.control_loop)
        
        rospy.loginfo("Control Node initialized. Cruise throttle: %.2f", self.cruise_throttle)

    def steering_callback(self, msg):
        self.lane_steering = msg.data

    def decision_callback(self, msg):
        if msg.decision != self.current_decision:
            rospy.loginfo("Control received decision: %s", msg.decision)
            self.current_decision = msg.decision
            self.decision_time = time.time()

    def control_loop(self, event):
        cmd = Twist()
        
        # Decision Command Mapping
        if self.current_decision == "STOP":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        elif self.current_decision == "CRUISE":
            # Normal bám làn
            cmd.linear.x = self.cruise_throttle
            cmd.angular.z = self.lane_steering
            
        elif self.current_decision == "TURN_LEFT":
            # Timed open-loop turn
            cmd.linear.x = self.turn_throttle
            cmd.angular.z = self.max_steer_left
            
        elif self.current_decision == "TURN_RIGHT":
            # Timed open-loop turn
            cmd.linear.x = self.turn_throttle
            cmd.angular.z = self.max_steer_right
            
        elif self.current_decision == "GO_STRAIGHT":
            # Ignore lane lines and drive straight forward
            cmd.linear.x = self.cruise_throttle
            cmd.angular.z = 0.0
            
        else:
            # Safe fallback
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        self.cmd_pub.publish(cmd)
