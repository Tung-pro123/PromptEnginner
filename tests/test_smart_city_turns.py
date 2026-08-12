#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Test Độc lập Chuyển hướng Ngã tư (Rẽ Trái, Rẽ Phải, Đi Thẳng, Đi Lùi) + WEB LIVE STREAM CAMERA.
Tích hợp Web Server phát trực tiếp Video Camera + Thống kê góc lái lên Trình duyệt máy tính.
"""

import sys
import os

# 1. Bắt buộc loại bỏ python2.7 trước để Python 3 import đúng cv2 và numpy gốc
sys.path = [p for p in sys.path if 'python2.7' not in p]

import cv2
import numpy as np
import time

# 2. Sau khi import cv2 & numpy xong mới nạp đường dẫn rospy
ros_paths = [
    "/opt/ros/melodic/lib/python2.7/dist-packages",
    "/media/jetson/ff2880cc-1a99-40bd-88c1-5cdc86fe9eed1/opt/ros/melodic/lib/python2.7/dist-packages"
]
for p in ros_paths:
    if p not in sys.path and os.path.exists(p):
        sys.path.append(p)

import rospy
from sensor_msgs.msg import Image

# Thêm đường dẫn thư mục gốc
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import settings
from src.core.control.racer_controller import RacerController
from src.smart_city.intersection_navigator import IntersectionNavigator, TurnAction
from src.debug.web_viewer import start_web_stream_server, set_web_frame

latest_frame = None

def camera_callback(msg):
    global latest_frame
    try:
        img = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == 'bgr8':
            latest_frame = img.reshape((msg.height, msg.width, 3))
        elif msg.encoding == 'rgb8':
            img_rgb = img.reshape((msg.height, msg.width, 3))
            latest_frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'mono8':
            latest_frame = cv2.cvtColor(img.reshape((msg.height, msg.width)), cv2.COLOR_GRAY2BGR)
    except Exception:
        pass

def main():
    rospy.init_node('test_smart_city_turns_node', anonymous=True)
    rospy.loginfo("=== KHỞI TẠO SCRIPT TEST CHUYỂN HƯỚNG NGÃ TƯ + WEB LIVE STREAM ===")

    rospy.Subscriber(settings.ROS_TOPIC_CAMERA, Image, camera_callback, queue_size=1)

    racer = RacerController()
    navigator = IntersectionNavigator()

    start_web_stream_server(port=8080)

    print("\n" + "="*60)
    print("  TRUY CẬP CAMERA TRỰC TIẾP TRÊN MÁY TÍNH:")
    print("    Mở trình duyệt Web (Chrome/Edge): http://<IP_CỦA_XE>:8080")
    print("-" * 60)
    print("  BẢNG LỆNH ĐIỀU KHIỂN CHẠY THỬ NGÃ TƯ & ĐI LÙI:")
    print("    - Nhấn 'l' hoặc 'left'    : Kích hoạt RẼ TRÁI")
    print("    - Nhấn 'r' hoặc 'right'   : Kích hoạt RẼ PHẢI")
    print("    - Nhấn 's' hoặc 'straight': Kích hoạt ĐI THẲNG")
    print("    - Nhấn 'b' hoặc 'back'    : Kích hoạt ĐI LÙI (REVERSE)")
    print("    - Nhấn 'q' hoặc 'quit'    : DỪNG XE VÀ THOÁT")
    print("="*60 + "\n")

    current_action = TurnAction.NONE

    try:
        while not rospy.is_shutdown():
            cmd = input("Nhập lệnh (l/r/s/b/q): ").strip().lower()

            if cmd in ['q', 'quit', 'exit']:
                rospy.loginfo("Dừng chương trình test.")
                break
            elif cmd in ['l', 'left']:
                current_action = TurnAction.LEFT
            elif cmd in ['r', 'right']:
                current_action = TurnAction.RIGHT
            elif cmd in ['s', 'straight']:
                current_action = TurnAction.STRAIGHT
            elif cmd in ['b', 'back', 'backward']:
                current_action = TurnAction.BACKWARD
            else:
                print("Lệnh không hợp lệ! Vui lòng nhập 'l', 'r', 's', 'b', hoặc 'q'.")
                continue

            rospy.loginfo(f"-> ĐANG THỰC THI HÀNH ĐỘNG: {current_action}...")
            start_time = time.time()
            
            while not rospy.is_shutdown():
                done = navigator.execute_turn(current_action, racer, camera_processor=None)

                if latest_frame is not None:
                    vis_frame = cv2.resize(latest_frame, (640, 480))
                    cv2.putText(vis_frame, f"ACTION: {current_action}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(vis_frame, f"STEER: {racer.car.steering if hasattr(racer.car, 'steering') else 0.0:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    set_web_frame(vis_frame)

                if done:
                    racer.stop()
                    print(f"[OK] Đã hoàn thành hành động {current_action} trong {time.time() - start_time:.2f} giây!\n")
                    break
                time.sleep(0.04)

    except KeyboardInterrupt:
        rospy.loginfo("Dừng do KeyboardInterrupt.")
    finally:
        racer.stop()
        rospy.loginfo("Đã dừng xe và giải phóng tài nguyên.")

if __name__ == '__main__':
    main()
