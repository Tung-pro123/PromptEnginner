import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from perception.camera.base_camera_processor import BaseCameraProcessor
from config import settings

class CameraProcessor(BaseCameraProcessor):
    def __init__(self, blackboard=None):
        self.blackboard = blackboard
        self.cap = None
        self.estimated_lane_width = 100.0 # Giá trị khởi tạo
        self.last_known_direction = 0.0
        self.latest_image = None

    def initialize(self):
        # Mở luồng GStreamer cho CSI camera hoặc USB camera
        # Tạm thời để cap = cv2.VideoCapture(0) cho mục đích test
        self.cap = cv2.VideoCapture(0)
        print("[INFO] Camera initialized.")

    def get_frame(self):
        if self.latest_image is not None:
            return self.latest_image
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def ros_callback(self, msg):
        """Chuyển đổi dữ liệu ảnh ROS Image thành numpy array OpenCV"""
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'bgr8':
                if self.blackboard:
                    self.blackboard.set('latest_image', img.reshape((msg.height, msg.width, 3)))
                else:
                    self.latest_image = img.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'rgb8':
                img_rgb = img.reshape((msg.height, msg.width, 3))
                if self.blackboard:
                    self.blackboard.set('latest_image', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                else:
                    self.latest_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                if self.blackboard:
                    self.blackboard.set('latest_image', img.reshape((msg.height, msg.width)))
                else:
                    self.latest_image = img.reshape((msg.height, msg.width))
        except Exception as e:
            print(f"Lỗi chuyển đổi ảnh: {e}")

    def process_frame(self, frame, dodge_direction=0.0):
        """
        Xử lý ảnh dựa trên Pipeline 3.1:
        1. Tiền xử lý
        2. Phân cụm trên nhiều hàng (scanlines)
        3. Tính toán waypoints
        """
        if frame is None:
            return settings.IMAGE_CENTER_X, []
            
        # 1. Tiền xử lý
        resized = cv2.resize(frame, (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, settings.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        y_lines = [160, 200, 240, 280]
        waypoints = []
        center_x_bottom = settings.IMAGE_CENTER_X
        
        for y in y_lines:
            scan_line = thresh[y, :]
            
            # 2. Phân cụm vạch
            white_pixels = np.where(scan_line == 255)[0]
            clusters = []
            if len(white_pixels) > 0:
                current_cluster = [white_pixels[0]]
                for i in range(1, len(white_pixels)):
                    if white_pixels[i] - white_pixels[i-1] <= settings.MAX_GAP_BETWEEN_POINTS:
                        current_cluster.append(white_pixels[i])
                    else:
                        clusters.append(int(np.mean(current_cluster)))
                        current_cluster = [white_pixels[i]]
                clusters.append(int(np.mean(current_cluster)))
                
            # 3. State-Aware Classification & Tính toán x
            center_x = settings.IMAGE_CENTER_X
            
            if len(clusters) >= 2:
                left_border = clusters[0]
                right_border = clusters[-1]
                center_x = (left_border + right_border) / 2.0
                
                if y == max(y_lines):
                    current_width = right_border - left_border
                    self.estimated_lane_width = 0.9 * self.estimated_lane_width + 0.1 * current_width
                
            elif len(clusters) == 1:
                line_pos = clusters[0]
                # State-Aware
                if dodge_direction == -1.0: # Đang né trái -> Vạch là biên trái
                    center_x = line_pos + (self.estimated_lane_width / 2.0)
                elif dodge_direction == 1.0: # Đang né phải -> Vạch là biên phải
                    center_x = line_pos - (self.estimated_lane_width / 2.0)
                else:
                    if line_pos < settings.IMAGE_CENTER_X:
                        center_x = line_pos + (self.estimated_lane_width / 2.0)
                    else:
                        center_x = line_pos - (self.estimated_lane_width / 2.0)
            else:
                center_x = settings.IMAGE_CENTER_X + (20 * self.last_known_direction)

            waypoints.append((int(center_x), y))
            if y == 240:
                center_x_bottom = center_x

        if center_x_bottom < settings.IMAGE_CENTER_X:
            self.last_known_direction = -1.0
        elif center_x_bottom > settings.IMAGE_CENTER_X:
            self.last_known_direction = 1.0

        return center_x_bottom, waypoints

    def process(self, blackboard):
        latest_image = blackboard.get('latest_image')
        dodge_direction = blackboard.get('dodge_direction', 0.0)
        center_x, waypoints = self.process_frame(latest_image, dodge_direction)
        blackboard.set('center_x', center_x)
        blackboard.set('lane_waypoints', waypoints)
