# -*- coding: utf-8 -*-
"""
WebCameraViewer: Trình phát luồng Camera thời gian thực qua Web Browser (Zero-dependency MJPEG Streamer).
Địa chỉ truy cập trên laptop: http://<IP_CỦA_XE>:8080
"""

import cv2
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

_latest_jpg = None
_lock = threading.Lock()

def set_web_frame(frame):
    global _latest_jpg
    if frame is None:
        return
    try:
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ret:
            with _lock:
                _latest_jpg = jpeg.tobytes()
    except Exception:
        pass

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class CamStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/', '/index.html']:
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Jetson AI Racer - Realtime Camera Stream</title>
                <style>
                    body { background-color: #121212; color: #ffffff; font-family: Arial, sans-serif; text-align: center; margin: 0; padding: 20px; }
                    h1 { color: #00e676; margin-bottom: 5px; }
                    .card { background: #1e1e1e; display: inline-block; padding: 15px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
                    img { width: 640px; height: 480px; border-radius: 8px; border: 2px solid #333; }
                    .info { margin-top: 10px; font-size: 14px; color: #bbb; }
                </style>
            </head>
            <body>
                <h1>🏎️ JETSON REALTIME CAMERA STREAM</h1>
                <div class="card">
                    <img src="/stream.mjpeg" />
                    <div class="info">Đang phát luồng Live Stream thời gian thực (Port 8080)</div>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/stream.mjpeg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                with _lock:
                    jpg = _latest_jpg
                if jpg is not None:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', str(len(jpg)))
                        self.end_headers()
                        self.wfile.write(jpg)
                        self.wfile.write(b'\r\n')
                    except Exception:
                        break
                time.sleep(0.04)
        else:
            self.send_error(404)

def start_web_stream_server(port=8080):
    server = ThreadedHTTPServer(('0.0.0.0', port), CamStreamHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    print(f"\n[WEB STREAM] >>> ĐÃ BẬT LIVE CAMERA STREAM TẠI: http://<IP_CỦA_XE>:{port}\n")
    return server
