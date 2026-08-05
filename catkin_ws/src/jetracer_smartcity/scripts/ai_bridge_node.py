#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import json
import numpy as np
import argparse

# Try imports for rospy and cv_bridge. They might fail if running in Py3 on a default ROS Melodic setup
ROSPY_AVAILABLE = False
try:
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from jetracer_smartcity.msg import Detection, DetectionArray
    from cv_bridge import CvBridge
    ROSPY_AVAILABLE = True
except ImportError:
    pass

from perception.detector import YoloDetector
from perception.traffic_light_state import TrafficLightClassifier

class AIBridgeNode:
    def __init__(self, use_zmq=False, zmq_port_sub=5555, zmq_port_pub=5556):
        self.use_zmq = use_zmq
        
        # Load Detector and Classifier
        self.detector = YoloDetector()
        self.light_classifier = TrafficLightClassifier()
        
        if self.use_zmq:
            import zmq
            self.context = zmq.Context()
            
            # Sub socket to receive raw frames from Python 2 bridge node
            self.sub_socket = self.context.socket(zmq.SUB)
            self.sub_socket.connect(f"tcp://localhost:{zmq_port_sub}")
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            
            # Pub socket to publish detection arrays back to Python 2 bridge node
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind(f"tcp://*:{zmq_port_pub}")
            
            print(f"[AIBridge] ZeroMQ Mode active. Sub: {zmq_port_sub}, Pub: {zmq_port_pub}")
        else:
            if not ROSPY_AVAILABLE:
                raise ImportError("rospy/CvBridge not available. Please run with --use_zmq")
                
            rospy.init_node('ai_bridge_node', anonymous=True)
            self.bridge = CvBridge()
            
            # ROS Publishers
            self.det_pub = rospy.Publisher('/detections', DetectionArray, queue_size=10)
            self.fps_pub = rospy.Publisher('/pipeline_fps', String, queue_size=10)
            
            # ROS Subscriber
            self.image_sub = rospy.Subscriber('/image_raw', Image, self.image_callback, queue_size=1)
            
            print("[AIBridge] Native ROS Python 3 Mode active.")

    def process_frame(self, frame):
        """
        Runs object detection and falls back to HSV color checking for traffic lights.
        Returns: list of detections, processing latency (ms)
        """
        # Run TensorRT/Mock YOLO detector
        detections, latency_ms = self.detector.infer(frame)
        
        # Verify color for any traffic lights using HSV color cropping (safety fallback)
        for det in detections:
            if det["label"] in ["red_light", "green_light"]:
                crop = self.light_classifier.crop_light_region(frame, det["bbox"])
                hsv_color = self.light_classifier.classify_color(crop)
                
                # Overwrite color if HSV classifier is confident
                if hsv_color in ["RED", "GREEN"]:
                    det["label"] = "red_light" if hsv_color == "RED" else "green_light"
                    
        return detections, latency_ms

    def image_callback(self, msg):
        """
        Callback used for native ROS mode.
        """
        try:
            # Convert ROS Image to OpenCV BGR frame
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn(f"Failed to convert image message: {e}")
            return
            
        start_time = rospy.Time.now()
        detections, latency_ms = self.process_frame(cv_img)
        
        # Construct and publish ROS message
        det_arr_msg = DetectionArray()
        det_arr_msg.header.stamp = start_time
        
        for det in detections:
            det_msg = Detection()
            det_msg.label = det["label"]
            det_msg.confidence = det["confidence"]
            det_msg.x, det_msg.y, det_msg.w, det_msg.h = det["bbox"]
            det_arr_msg.detections.append(det_msg)
            
        self.det_pub.publish(det_arr_msg)
        
        # Publish latency/FPS data
        fps = 1000.0 / latency_ms if latency_ms > 0 else 30.0
        self.fps_pub.publish(String(f"{fps:.2f}"))

    def run_zmq_loop(self):
        """
        Blocking loop used for ZeroMQ bridge mode.
        """
        print("[AIBridge] Starting ZeroMQ loop...")
        try:
            while True:
                # Receive serialized image frame
                message = self.sub_socket.recv()
                
                # Image is received as jpeg byte buffer
                np_arr = np.frombuffer(message, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                    
                # Process
                detections, latency_ms = self.process_frame(frame)
                
                # Serialize detections to JSON and publish
                payload = {
                    "latency_ms": latency_ms,
                    "detections": detections
                }
                self.pub_socket.send_string(json.dumps(payload))
                
        except KeyboardInterrupt:
            print("[AIBridge] Stopping ZeroMQ bridge...")
        finally:
            self.sub_socket.close()
            self.pub_socket.close()
            self.context.term()

    def spin(self):
        if self.use_zmq:
            self.run_zmq_loop()
        else:
            rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JetRacer AI Bridge Node")
    parser.add_argument("--use_zmq", action="store_true", help="Use ZeroMQ bridge mode")
    parser.add_argument("--sub_port", type=int, default=5555, help="ZMQ Subscriber port")
    parser.add_argument("--pub_port", type=int, default=5556, help="ZMQ Publisher port")
    
    # If ROS is executing, arguments might be appended by roslaunch
    # Filter args to keep parser happy
    args, unknown = parser.parse_known_args()
    
    bridge_node = AIBridgeNode(
        use_zmq=args.use_zmq, 
        zmq_port_sub=args.sub_port, 
        zmq_port_pub=args.pub_port
    )
    bridge_node.spin()
