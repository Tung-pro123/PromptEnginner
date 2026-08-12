#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Test TỰ ĐỘNG CHUYỂN HƯỚNG DỰA TRÊN CAMERA (Pure Vision Automatic Turning) + WEB LIVE STREAM.
Tự động phát hiện Dải Vạch Trắng Ngang Ngã Tư -> Tự Nhô đầu xe 35cm qua khung nhôm -> Tự Bẻ lái ôm cua -> Tự Khóa làn!

Truy cập Web Stream trên Máy tính:
    http://<IP_CỦA_XE>:8080

Chạy lệnh trên xe:
    python3 tests/test_auto_smart_city.py
"""

import sys
import os
import time

# Ưu tiên Python 3, đẩy python2.7 ra sau tránh xung đột
py3 = [p for p in sys.path if 'python2.7' not in p]
py2 = [p for p in sys.path if 'python2.7' in p]
sys.path = py3 + py2

if 'enum' in sys.modules and not hasattr(sys.modules['enum'], 'IntFlag'):
    del sys.modules['enum']

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

# Thêm đường dẫn thư mục gốc
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import settings
from src.core.control.racer_controller import RacerController
from src.smart_city.auto_turn_controller import AutoTurnController, AutoTurnState
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
    rospy.init_node('test_auto_smart_city_node', anonymous=True)
    rospy.loginfo("=== KHỞI TẠO SCRIPT TEST TỰ ĐỘNG NGÃ TƯ BẰNG CAMERA (PURE VISION) ===")

    rospy.Subscriber(settings.ROS_TOPIC_CAMERA, Image, camera_callback, queue_size=1)

    racer = RacerController()
    auto_turn = AutoTurnController()

    start_web_stream_server(port=8080)

    print("\n" + "="*60)
    print("  TRUY CẬP LIVE CAMERA TRÊN MÁY TÍNH:")
    print("    Mở trình duyệt Web (Chrome/Edge): http://<IP_CỦA_XE>:8080")
    print("-" * 60)
    print("  SẴN SÀNG TỰ ĐỘNG CHẠY & TỰ RẼ NGÃ TƯ THỊ GIÁC!")
    print("    - Nhấn Ctrl + C để dừng xe bất kỳ lúc nào.")
    print("="*60 + "\n")

    rate = rospy.Rate(20)

    try:
        while not rospy.is_shutdown():
            if latest_frame is None:
                rate.sleep()
                continue

            frame = latest_frame.copy()
            state_name, is_handling = auto_turn.update(frame, racer)

            if not is_handling:
                if hasattr(racer, 'steer'):
                    racer.steer(0.0, 0.18)
                elif hasattr(racer, 'set_steering'):
                    racer.set_steering(0.0)
                    if hasattr(racer, 'set_throttle'):
                        racer.set_throttle(0.18)

            vis = cv2.resize(frame, (640, 480))
            cv2.putText(vis, f"AUTO STATE: {state_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            set_web_frame(vis)

            rate.sleep()

    except KeyboardInterrupt:
        rospy.loginfo("Dừng do KeyboardInterrupt.")
    finally:
        racer.stop()
        rospy.loginfo("Đã dừng xe và giải phóng tài nguyên.")

if __name__ == '__main__':
    main()
