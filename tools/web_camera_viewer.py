#!/usr/bin/env python3
"""
Web Live Camera Streamer for Jetson Nano
Tạo Web Server nhẹ truyền luồng video thời gian thực từ ROS topic /csi_cam_0/image_raw về trình duyệt Laptop.

Cách dùng:
  1. Chạy trên Jetson: python3 tools/web_camera_viewer.py
  2. Mở trình duyệt trên Laptop: http://192.168.55.1:8080 (hoặc IP WiFi của xe:8080)
"""

import sys
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

latest_jpeg = None
lock = threading.Lock()
fps_counter = 0
fps_display = 0.0
last_time = time.time()

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <title>JetRacer CSI Camera Live Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
        }
        h1 {
            color: #38bdf8;
            margin-bottom: 5px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: #1e293b;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }
        .video-box {
            position: relative;
            display: inline-block;
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid #334155;
        }
        img {
            display: block;
            max-width: 100%;
            height: auto;
        }
        .badge {
            display: inline-block;
            background: #0284c7;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            margin-top: 10px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏎️ JetRacer CSI Camera Live</h1>
        <p style="color: #94a3b8; margin-top: 0;">Luồng phát thời gian thực từ /csi_cam_0/image_raw</p>
        <div class="video-box">
            <img src="/stream.mjpg" alt="Đang kết nối luồng Camera...">
        </div>
        <br>
        <div class="badge">Trạng thái: Đang phát trực tiếp</div>
    </div>
</body>
</html>
"""

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_jpeg, lock
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with lock:
                        if latest_jpeg is not None:
                            jpeg_bytes = latest_jpeg
                        else:
                            jpeg_bytes = None
                    
                    if jpeg_bytes is not None:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(jpeg_bytes)))
                        self.end_headers()
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.033)  # ~30 FPS
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()

def image_callback(msg):
    global latest_jpeg, lock, fps_counter, fps_display, last_time
    try:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        fps_counter += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps_display = fps_counter / (now - last_time)
            fps_counter = 0
            last_time = now
            
        # Vẽ thông số lên góc ảnh
        cv2.putText(img, f"FPS: {fps_display:.1f} | {msg.width}x{msg.height}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, time.strftime("%H:%M:%S"), (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Nén JPEG chất lượng 80 để stream mượt mà qua mạng
        _, encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        with lock:
            latest_jpeg = encoded.tobytes()
    except Exception as e:
        rospy.logerr_throttle(5, f"Lỗi nén frame: {e}")

def main():
    rospy.init_node('web_camera_viewer', anonymous=True)
    rospy.Subscriber('/csi_cam_0/image_raw', Image, image_callback, queue_size=1)
    
    port = 8080
    server = HTTPServer(('0.0.0.0', port), StreamHandler)
    rospy.loginfo(f"🌐 Web Streamer đã chạy tại port {port}!")
    rospy.loginfo(f"👉 Mở trình duyệt trên máy tính tại: http://192.168.55.1:{port}")
    
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    rospy.spin()

if __name__ == '__main__':
    main()
