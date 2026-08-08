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
            self.use_advanced_segmentation = getattr(settings, 'USE_ADVANCED_SEGMENTATION', False)
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
        Xử lý ảnh bằng lọc màu HSV Cam/Đỏ để trích xuất vạch ranh giới, 
        sau đó dùng Contours lọc nhiễu và quét hàng (Scanline) tìm tâm đường đen ở giữa.
        """
        if frame is None:
            return settings.IMAGE_CENTER_X, [], None
            
        # 1. Tiền xử lý bằng hệ màu HSV để chống nhiễu ánh sáng và vết bẩn lòng đường
        resized = cv2.resize(frame, (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Ngưỡng màu Đỏ/Cam (Dải 1: Đỏ nhạt đến cam)
        lower_red1 = np.array([0, 110, 50])
        upper_red1 = np.array([22, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        # Ngưỡng màu Đỏ/Cam (Dải 2: Đỏ đậm)
        lower_red2 = np.array([160, 110, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Hợp nhất dải màu để lấy trọn vẹn vạch biên màu cam/đỏ
        thresh = cv2.bitwise_or(mask1, mask2)
        
        # 1.4 Khép kín các lỗ đứt gãy nhỏ trên vạch kẻ bằng phép toán đóng (Closing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 1.5 Dùng Contours để lọc bỏ các vùng đốm trắng nhiễu nhỏ (Tương thích cả OpenCV 3 và OpenCV 4)
        contours_data = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
        valid_contours = [c for c in contours if cv2.contourArea(c) > 30]  # Giữ lại vạch ranh giới thực sự
        
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
        # Thu hẹp cự ly quét và dùng 8 điểm cách nhau 5px
        y_lines = [180, 185, 190, 195, 200, 205, 210, 215]
        waypoints = []
        raw_waypoints = []
        road_boundaries = {}
        center_x_bottom = settings.IMAGE_CENTER_X
        
        # Độ rộng cửa sổ kiểm tra (window size) để lọc nhiễu đốm nhỏ
        # Một điểm chuyển giao chỉ được coi là biên nếu đó là vạch màu trắng (255) rộng ít nhất 4 pixel
        check_w = 4
        
        for y in y_lines:
            scan_line = clean_thresh[y, :]
            
            # Quét tìm ranh giới biên bên trái (tìm vạch ranh giới màu trắng 255)
            found_left = False
            left_border = 0
            for x in range(1, settings.IMAGE_CENTER_X - check_w):
                if scan_line[x] == 255:
                    if np.all(scan_line[x : x + check_w] == 255):
                        left_border = x + check_w // 2
                        found_left = True
                        break
                    
            # Quét tìm ranh giới biên bên phải (tìm vạch ranh giới màu trắng 255 từ phải qua trái)
            found_right = False
            right_border = settings.IMAGE_WIDTH - 1
            for x in range(settings.IMAGE_WIDTH - 2, settings.IMAGE_CENTER_X + check_w, -1):
                if scan_line[x] == 255:
                    if np.all(scan_line[x - check_w + 1 : x + 1] == 255):
                        right_border = x - check_w // 2
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
        # Vẽ viền vùng quan tâm ROI bằng nét màu xanh lá cây mỏng
        cv2.polylines(debug_img, [roi_corners], True, (0, 255, 0), 1)
        return center_x_bottom, waypoints, debug_img

    def process(self, blackboard):
        latest_image = blackboard.get('latest_image')
        dodge_direction = blackboard.get('dodge_direction', 0.0)
        
        # Lưu frame gốc chưa xử lý để debugger ghi vào raw_camera.avi
        # (phải lưu trước khi latest_image bị ghi đè bởi debug_img)
        if latest_image is not None:
            blackboard.set('raw_camera_frame', latest_image)
        
        # =====================================================================
        # NHÁNH 1: Thuật toán Boundary Following mới (detect_boundary_path)
        # Ưu tiên cao nhất khi USE_BOUNDARY_PATH = True
        # =====================================================================
        if getattr(settings, 'USE_BOUNDARY_PATH', False) and self.lane_detector is not None:
            boundary_offset_px = getattr(settings, 'BOUNDARY_OFFSET_PX', 55)
            center_x, waypoints, debug_img, bev_debug_img = self.lane_detector.detect_boundary_path(
                latest_image,
                boundary_offset_px=boundary_offset_px,
                debug=True
            )
            
            blackboard.set('center_x', center_x)
            blackboard.set('lane_waypoints', waypoints)
            
            # Ghi ảnh debug gốc (có overlay waypoints)
            if debug_img is not None:
                blackboard.set('latest_image', debug_img)
            # Ghi thêm ảnh Bird's Eye View cho test script hiển thị
            if bev_debug_img is not None:
                blackboard.set('bev_debug_img', bev_debug_img)
                blackboard.set('camera_thresh', cv2.cvtColor(bev_debug_img, cv2.COLOR_BGR2GRAY))
                
        # =====================================================================
        # NHÁNH 2: Thuật toán cũ - HSV Sliding Window (USE_ADVANCED_SEGMENTATION)
        # =====================================================================
        elif getattr(self, 'use_advanced_segmentation', False) and self.lane_detector is not None:
            # 1. Sử dụng thuật toán phân đoạn ảnh (Sliding Window & Curve Fitting)
            if getattr(settings, 'USE_COLOR_SEGMENTATION', False):
                # Sử dụng lọc màu HSV + Sliding Window & Curve Fitting
                segmented_img, thresh, center_x, left_fit, right_fit = self.lane_detector.process_color_segment(latest_image)
            else:
                # Sử dụng Grayscale threshold + Sliding Window & Curve Fitting
                segmented_img, thresh, center_x, left_fit, right_fit = self.lane_detector.process_and_segment(latest_image, settings.THRESHOLD_VALUE)
            
            # 2. Sinh ra danh sách waypoints để tương thích ngược với PredictiveController
            waypoints = []
            if left_fit is not None and right_fit is not None:
                # Dùng 8 điểm quét với khoảng cách hẹp hơn (từ 180 đến 285, cách nhau 15px)
                y_lines = [180, 195, 210, 225, 240, 255, 270, 285]
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
        # =====================================================================
        # NHÁNH 3: Fallback - Pipeline đơn giản (Scanline quét hàng)
        # =====================================================================
        else:
            # Chạy thuật toán dự phòng (Pipeline cũ, đơn giản)
            center_x, waypoints, thresh = self.process_frame(latest_image, dodge_direction)
            
            blackboard.set('center_x', center_x)
            blackboard.set('lane_waypoints', waypoints)
            if thresh is not None:
                blackboard.set('camera_thresh', thresh)

