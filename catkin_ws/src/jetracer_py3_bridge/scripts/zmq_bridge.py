#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import json
import numpy as np
import zmq
from sensor_msgs.msg import Image
from std_msgs.msg import String
from jetracer_smartcity.msg import Detection, DetectionArray

class ZmqBridgeNode:
    def __init__(self):
        rospy.init_node('zmq_bridge', anonymous=True)
        
        # Configuration
        self.pub_port = rospy.get_param('~pub_port', 5555) # Port to send images to Py3
        self.sub_port = rospy.get_param('~sub_port', 5556) # Port to receive detections from Py3
        
        # Initialize ZeroMQ
        self.context = zmq.Context()
        
        # Pub socket: send compressed images to Py3
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:" + str(self.pub_port))
        
        # Sub socket: receive detections from Py3
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect("tcp://localhost:" + str(self.sub_port))
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Setup cv_bridge equivalent to prevent Python version clashes
        # Convert ROS sensor_msgs/Image directly using numpy buffer (very fast)
        self.encoding_map = {
            'bgr8': cv2.IMREAD_COLOR,
            'rgb8': cv2.IMREAD_COLOR
        }
        
        # ROS Publishers
        self.det_pub = rospy.Publisher('/detections', DetectionArray, queue_size=10)
        self.fps_pub = rospy.Publisher('/pipeline_fps', String, queue_size=10)
        
        # ROS Subscriber
        self.img_sub = rospy.Subscriber('/image_raw', Image, self.image_callback, queue_size=1)
        
        # Non-blocking ZMQ check using a timer (30Hz)
        self.timer = rospy.Timer(rospy.Duration(0.033), self.check_zmq_messages)
        
        rospy.loginfo("ZeroMQ ROS Bridge active. Image Pub: %d, Detection Sub: %d", self.pub_port, self.sub_port)

    def image_callback(self, msg):
        """
        Receives ROS image, compresses to JPEG, and forwards it to Python 3.
        """
        try:
            # Parse raw byte array directly to avoid cv_bridge dependencies
            if msg.encoding == 'bgr8':
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'rgb8':
                frame_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                rospy.logwarn("Unsupported encoding format: %s", msg.encoding)
                return
                
            # Compress image to jpeg to reduce socket bandwidth
            _, jpeg_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            # Send binary jpeg frame over ZMQ socket
            self.pub_socket.send(jpeg_buffer.tobytes())
            
        except Exception as e:
            rospy.logwarn("ZMQ Bridge image conversion error: %s", str(e))

    def check_zmq_messages(self, event):
        """
        Checks for returned detection messages from Python 3 ZMQ socket.
        """
        try:
            # Check if there is data pending in ZMQ buffer without blocking ROS thread
            message = self.sub_socket.recv_string(flags=zmq.NOBLOCK)
            
            payload = json.loads(message)
            detections = payload.get("detections", [])
            latency_ms = payload.get("latency_ms", 0.0)
            
            # Construct ROS message
            det_arr_msg = DetectionArray()
            det_arr_msg.header.stamp = rospy.Time.now()
            
            for det in detections:
                det_msg = Detection()
                det_msg.label = det["label"]
                det_msg.confidence = det["confidence"]
                det_msg.x, det_msg.y, det_msg.w, det_msg.h = det["bbox"]
                det_arr_msg.detections.append(det_msg)
                
            self.det_pub.publish(det_arr_msg)
            
            # Publish FPS statistics
            fps = 1000.0 / latency_ms if latency_ms > 0 else 30.0
            self.fps_pub.publish(String(str(round(fps, 2))))
            
        except zmq.Again:
            # No message in buffer, continue normal operations
            pass
        except Exception as e:
            rospy.logwarn("Error parsing ZMQ message: %s", str(e))

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        bridge = ZmqBridgeNode()
        bridge.spin()
    except rospy.ROSInterruptException:
        pass
