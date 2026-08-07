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
        
        # EMA Filter cho waypoints
        self.ema_waypoints = {}
        self.ema_alpha = 0.6  # Hệ số làm mượt (0-1). 1.0 = không làm mượt, 0.0 = giữ nguyên quá khứ.
        
        # Tích hợp LaneDetector (Phân đoạn ảnh - Computer Vision)
        try:
            from src.perception.camera.detect_lane import LaneDetector
            self.lane_detector = LaneDetector(image_width=settings.IMAGE_WIDTH, image_height=settings.IMAGE_HEIGHT)
            self.use_advanced_segmentation = False # Bỏ phân đoạn nâng cao màu xanh lá, chạy Hough Transform ở nhánh else
        except ImportError as e:
            print(f"[WARN] Không thể import LaneDetector: {e}")
            self.lane_detector = None
            self.use_advanced_segmentation = False

    def initialize(self):
        # Mở luồng GStreamer cho CSI camera hoặc USB camera
        # Tạm thời để cap = cv2.VideoCapture(0) cho mục đích test
        # self.cap = cv2.VideoCapture(0)
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
        import rospy
        rospy.loginfo_throttle(1.0, "[DEBUG] CameraProcessor: ĐÃ NHẬN được frame ảnh từ ROS Topic!")
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
        Xử lý ảnh kết hợp Threshold + Contours lọc nhiễu + Canny tìm biên + Phân cụm quét hàng.
        """
        if frame is None:
            return settings.IMAGE_CENTER_X, [], None
            
        # 1. Tiền xử lý
        resized = cv2.resize(frame, (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 1.1 Lọc nhiễu bằng Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 1.2 Cân bằng histogram cục bộ (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # 1.3 Nhị phân hóa tự động bằng thuật toán Otsu
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 1.4 Khép kín các lỗ đứt gãy nhỏ trên vạch kẻ bằng phép toán đóng (Closing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 1.5 Dùng Contours để lọc bỏ các vùng đốm trắng nhiễu nhỏ (Tương thích cả OpenCV 3 và OpenCV 4)
        contours_data = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
        valid_contours = [c for c in contours if cv2.contourArea(c) > 60]  # Lọc bỏ nhiễu diện tích < 60 px^2
        
        # Tạo ảnh mask sạch chỉ chứa các vạch kẻ đường hợp lệ
        clean_thresh = np.zeros_like(thresh_closed)
        cv2.drawContours(clean_thresh, valid_contours, -1, 255, -1)
        
        # 2. Canny Edge Detection trên ảnh đã sạch nhiễu hoàn toàn
        edges = cv2.Canny(clean_thresh, 50, 150)
        
        # Mask vùng quan tâm (ROI) - Nửa dưới ảnh
        mask = np.zeros_like(edges)
        roi_corners = np.array([[(0, settings.IMAGE_HEIGHT), 
                                 (0, settings.IMAGE_HEIGHT // 2), 
                                 (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT // 2), 
                                 (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Ghi nhận ảnh Canny sạch vào blackboard để phục vụ debug ghi file riêng
        if self.blackboard is not None:
            self.blackboard.set('canny_edges', masked_edges)
            
        # 3. Phân cụm quét hàng (Scanline) tìm biên đường đen (phương án C)
        y_lines = [160, 200, 240, 280]
        waypoints = []
        raw_waypoints = []
        road_boundaries = {}
        center_x_bottom = settings.IMAGE_CENTER_X
        
        for y in y_lines:
            scan_line = clean_thresh[y, :]
            
            # Quét tìm ranh giới biên bên trái (chuyển đổi từ nền trắng 255 sang lòng đường đen 0)
            found_left = False
            left_border = 0
            for x in range(1, settings.IMAGE_CENTER_X):
                if scan_line[x - 1] == 255 and scan_line[x] == 0:
                    left_border = x
                    found_left = True
                    break
                    
            # Quét tìm ranh giới biên bên phải (chuyển từ nền trắng 255 sang lòng đường đen 0 từ phải qua trái)
            found_right = False
            right_border = settings.IMAGE_WIDTH - 1
            for x in range(settings.IMAGE_WIDTH - 2, settings.IMAGE_CENTER_X, -1):
                if scan_line[x + 1] == 255 and scan_line[x] == 0:
                    right_border = x
                    found_right = True
                    break
            
            # Lưu biên thực tế quét được (nếu không thấy thì xem như mép ảnh)
            road_boundaries[y] = (left_border, right_border)
            
            center_x = settings.IMAGE_CENTER_X
            
            if found_left and found_right:
                # Nhìn thấy cả 2 biên ranh giới trắng trái/phải
                center_x = (left_border + right_border) / 2.0
                if y == max(y_lines):
                    current_width = right_border - left_border
                    if 160 < current_width < 280:
                        self.estimated_lane_width = 0.9 * self.estimated_lane_width + 0.1 * current_width
            elif found_left:
                # Chỉ thấy biên trái
                center_x = left_border + (self.estimated_lane_width / 2.0)
            elif found_right:
                # Chỉ thấy biên phải
                center_x = right_border - (self.estimated_lane_width / 2.0)
            else:
                # Mất cả 2 biên -> Rà mù theo hướng bẻ lái gần nhất
                center_x = settings.IMAGE_CENTER_X + (20 * self.last_known_direction)
                
            # Lưu điểm trung tâm chưa làm mượt
            raw_waypoints.append((int(center_x), y))
            
            # Làm mượt bằng EMA (Exponential Moving Average)
            if y not in self.ema_waypoints:
                self.ema_waypoints[y] = center_x
            else:
                self.ema_waypoints[y] = self.ema_alpha * center_x + (1.0 - self.ema_alpha) * self.ema_waypoints[y]
                
            smoothed_center_x = self.ema_waypoints[y]
            waypoints.append((int(smoothed_center_x), y))
            
            if y == 240:
                center_x_bottom = smoothed_center_x
                
        if self.blackboard is not None:
            self.blackboard.set('raw_waypoints', raw_waypoints)
            self.blackboard.set('road_boundaries', road_boundaries)
            
        if center_x_bottom < settings.IMAGE_CENTER_X:
            self.last_known_direction = -1.0
        elif center_x_bottom > settings.IMAGE_CENTER_X:
            self.last_known_direction = 1.0
            
        # Trực quan hóa ảnh Canny sạch cho khối Debugger hiển thị
        debug_img = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
        return center_x_bottom, waypoints, debug_img

    def process(self, blackboard):
        latest_image = blackboard.get('latest_image')
        dodge_direction = blackboard.get('dodge_direction', 0.0)
        
        if getattr(self, 'use_advanced_segmentation', False) and self.lane_detector is not None:
            # 1. Sử dụng thuật toán phân đoạn ảnh (Sliding Window & Curve Fitting)
            segmented_img, thresh, center_x, left_fit, right_fit = self.lane_detector.process_and_segment(latest_image, settings.THRESHOLD_VALUE)
            
            # 2. Sinh ra danh sách waypoints để tương thích ngược với PredictiveController
            waypoints = []
            if left_fit is not None and right_fit is not None:
                y_lines = [160, 200, 240, 280]
                for y in y_lines:
                    lx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
                    rx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
                    waypoints.append((int((lx + rx) / 2.0), y))
            
            # 3. Ghi dữ liệu vào Blackboard
            blackboard.set('center_x', center_x)
            blackboard.set('lane_waypoints', waypoints)
            if thresh is not None:
                blackboard.set('camera_thresh', thresh)
            if segmented_img is not None:
                # Ghi đè biến latest_image bằng ảnh đã phân đoạn để Debugger hiển thị dải màu xanh lá
                blackboard.set('latest_image', segmented_img)
        else:
            # Chạy thuật toán dự phòng (Pipeline cũ, đơn giản)
            center_x, waypoints, thresh = self.process_frame(latest_image, dodge_direction)
            
            blackboard.set('center_x', center_x)
            blackboard.set('lane_waypoints', waypoints)
            if thresh is not None:
                blackboard.set('camera_thresh', thresh)
