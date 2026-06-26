#!/usr/bin/env python3
import os
import time
import json
import cv2
import rospy
import requests
import numpy as np
from sensor_msgs.msg import Image

# Config (có thể override bằng biến môi trường)
RF_MODEL = os.environ.get('RF_MODEL', "dataset3-c4kyj/1")
RF_VERSION = os.environ.get('RF_VERSION', "1")
RF_API_KEY = os.environ.get('ROBOFLOW_API_KEY', "")
DETECT_INTERVAL = float(os.environ.get('CAM_CHECK_INTERVAL', "1.0"))  # giây giữa 2 lần detect

class CameraChecker:
    def __init__(self, topic='/csi_cam_0/image_raw'):
        rospy.init_node('camera_check_node', anonymous=True)
        self.topic = topic
        self.latest_image = None
        self.last_detect = 0.0
        rospy.Subscriber(self.topic, Image, self.image_cb)
        # auto-detect if DISPLAY available; nếu không thì chạy headless
        self.headless = os.environ.get('DISPLAY') is None
        if self.headless:
            rospy.loginfo("[camera_check] No DISPLAY detected -> running in headless mode (no cv2.imshow).")
            # thư mục lưu ảnh tạm
            self._out_dir = os.path.join('/tmp', 'camera_check_out')
            try:
                os.makedirs(self._out_dir, exist_ok=True)
            except Exception:
                self._out_dir = '.'
        else:
            rospy.loginfo(f"[camera_check] Subscribed to {self.topic} (DISPLAY present).")
        rospy.loginfo(f"[camera_check] Subscribed to {self.topic}")

    def image_cb(self, msg):
        try:
            if hasattr(msg, 'data') and msg.data is not None and msg._connection_header.get('type','').endswith('CompressedImage'):
                # unlikely here, but keep safe
                np_arr = np.frombuffer(msg.data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                # raw image: convert from buffer
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                channels = 3
                if msg.encoding and 'rgb' in msg.encoding.lower():
                    channels = 3
                # reshape may fail if message layout different - try best effort
                try:
                    frame = arr.reshape((msg.height, msg.width, -1))
                except Exception:
                    frame = arr.copy()
            # convert RGB->BGR if needed
            if msg.encoding and 'rgb' in msg.encoding.lower():
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.latest_image = cv2.resize(frame, (300,300))
        except Exception as e:
            rospy.logwarn(f"[camera_check] Failed to convert image: {e}")

    def roboflow_detect(self, image):
        if image is None:
            return []
        if not RF_API_KEY:
            rospy.logwarn("[camera_check] ROBOFLOW_API_KEY not set; skipping detect.")
            return []
        try:
            _, img_encoded = cv2.imencode('.jpg', image)
            files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
            url = f"https://detect.roboflow.com/{RF_MODEL}/{RF_VERSION}"
            params = {'api_key': RF_API_KEY}
            r = requests.post(url, params=params, files=files, timeout=6)
            if r.status_code != 200:
                rospy.logwarn(f"[camera_check] Roboflow returned {r.status_code}: {r.text}")
                return []
            j = r.json()
            preds = j.get('predictions', [])
            detections = []
            h, w = image.shape[:2]
            for p in preds:
                cx = p.get('x'); cy = p.get('y'); pw = p.get('width'); ph = p.get('height')
                if 0 < cx <= 1 and 0 < cy <= 1 and 0 < pw <= 1 and 0 < ph <= 1:
                    cx *= w; cy *= h; pw *= w; ph *= h
                x1 = int(cx - pw/2); y1 = int(cy - ph/2); x2 = int(cx + pw/2); y2 = int(cy + ph/2)
                detections.append({
                    'class_name': str(p.get('class', p.get('label','unknown'))),
                    'confidence': float(p.get('confidence', 0.0)),
                    'box': [max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)]
                })
            return detections
        except Exception as e:
            rospy.logerr(f"[camera_check] Detect error: {e}")
            return []

    def draw_and_show(self, img, detections):
        disp = img.copy()
        for d in detections:
            x1,y1,x2,y2 = d['box']
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
            label = f"{d['class_name']}:{d['confidence']:.2f}"
            cv2.putText(disp, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        if self.headless:
            # lưu ảnh annotate để kiểm tra (mỗi lần ghi timestamp)
            try:
                fname = f"detect_{int(time.time()*1000)}.jpg"
                path = os.path.join(self._out_dir, fname)
                cv2.imwrite(path, disp)
                rospy.loginfo(f"[camera_check] Headless mode: saved annotated image to {path}")
            except Exception as e:
                rospy.logwarn(f"[camera_check] Failed to save annotated image: {e}")
        else:
            cv2.imshow("camera_check", disp)
            cv2.waitKey(1)

    def loop(self):
        rate = rospy.Rate(10)
        rospy.loginfo("[camera_check] Ready. Press Ctrl-C to exit. Press 'd' in window to force detect now.")
        while not rospy.is_shutdown():
            img = self.latest_image
            if img is not None:
                now = time.time()
                # periodic detect
                if now - self.last_detect >= DETECT_INTERVAL:
                    detections = self.roboflow_detect(img)
                    # log detection results to ROS (on-screen)
                    try:
                        rospy.loginfo("[camera_check] Detections: " + json.dumps(detections, ensure_ascii=False))
                    except Exception:
                        rospy.loginfo(f"[camera_check] Detections: {detections}")
                    self.draw_and_show(img, detections)
                    self.last_detect = now
                else:
                    # Nếu không headless thì hiển thị, vẫn cho phép phím d/q
                    if not self.headless:
                        cv2.imshow("camera_check", img)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('d'):
                            detections = self.roboflow_detect(img)
                            try:
                                rospy.loginfo("[camera_check] Detections (manual): " + json.dumps(detections, ensure_ascii=False))
                            except Exception:
                                rospy.loginfo(f"[camera_check] Detections (manual): {detections}")
                            self.draw_and_show(img, detections)
                            self.last_detect = time.time()
                        elif key == ord('q'):
                            rospy.signal_shutdown("User requested quit")
                            break
                    else:
                        # headless: nhỏ ngủ, không gọi waitKey
                        time.sleep(0.01)
            rate.sleep()
        if not self.headless:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        checker = CameraChecker()
        checker.loop()
    except rospy.ROSInterruptException:
        pass