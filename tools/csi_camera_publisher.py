#!/usr/bin/env python3
"""
CSI Camera ROS Publisher Node for Jetson Nano
Sử dụng trực tiếp GStreamer fdsink qua Linux Pipe để bắn ảnh lên ROS topic /csi_cam_0/image_raw.
Tối ưu hóa độ trễ thấp nhất (Zero latency) và tự động giải phóng tài nguyên nvargus-daemon.
"""

import sys
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

import rospy
from sensor_msgs.msg import Image
import subprocess
import numpy as np
import time
import signal
import os

WIDTH = 640
HEIGHT = 480
FPS = 30
CHANNELS = 3
FRAME_BYTES = WIDTH * HEIGHT * CHANNELS

def main():
    rospy.init_node('csi_camera_publisher', anonymous=False)
    pub = rospy.Publisher('/csi_cam_0/image_raw', Image, queue_size=1)
    
    # Dọn sạch các tiến trình gst-launch bị treo trước đó
    os.system("pkill -9 gst-launch-1.0 2>/dev/null")
    time.sleep(0.5)
    
    sensor_id = rospy.get_param('~sensor_id', 0)
    
    pipeline_cmd = [
        'gst-launch-1.0', '-q',
        'nvarguscamerasrc', f'sensor-id={sensor_id}', '!',
        'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1', '!',
        'nvvidconv', '!',
        f'video/x-raw, width={WIDTH}, height={HEIGHT}, format=BGRx', '!',
        'videoconvert', '!',
        'video/x-raw, format=BGR', '!',
        'fdsink'
    ]
    
    rospy.loginfo(f"Khởi động luồng Camera CSI GStreamer sensor-id={sensor_id} ({WIDTH}x{HEIGHT} @ {FPS} FPS)...")
    
    proc = subprocess.Popen(pipeline_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=FRAME_BYTES * 2)
    
    def cleanup_handler(signum, frame):
        rospy.loginfo("Đang dừng luồng camera...")
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            proc.kill()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    img_msg = Image()
    img_msg.header.frame_id = "csi_cam_0_link"
    img_msg.height = HEIGHT
    img_msg.width = WIDTH
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = False
    img_msg.step = WIDTH * CHANNELS
    
    frame_count = 0
    start_time = time.time()
    
    rospy.loginfo("✅ Camera CSI đã sẵn sàng phát lên topic /csi_cam_0/image_raw!")
    
    try:
        while not rospy.is_shutdown():
            raw_data = proc.stdout.read(FRAME_BYTES)
            if len(raw_data) != FRAME_BYTES:
                if proc.poll() is not None:
                    err = proc.stderr.read().decode('utf-8', errors='ignore')
                    rospy.logerr(f"GStreamer process terminated: {err}")
                    break
                continue
                
            img_msg.header.stamp = rospy.Time.now()
            img_msg.data = raw_data
            pub.publish(img_msg)
            
            frame_count += 1
            if frame_count % 150 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                rospy.loginfo(f"Camera CSI đang phát: {fps_actual:.1f} FPS (Frame #{frame_count})")
                
    except KeyboardInterrupt:
        pass
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            proc.kill()


if __name__ == '__main__':
    main()
