import csv
import time
import os
import math
import cv2
import numpy as np

class Debugger:
    """Lớp tiện ích để hỗ trợ debug chương trình."""

    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.csv_file = None
        self.csv_writer = None
        self.cam_writer = None
        self.lidar_writer = None
        self.info_path = None
        
        if self.debug_mode:
            import glob
            import datetime
            # Khởi tạo thư mục log cơ sở
            repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_base_dir = os.path.join(repo_dir, "logs")
            if not os.path.exists(log_base_dir):
                os.makedirs(log_base_dir)
                
            # Xác định session tiếp theo
            existing_sessions = glob.glob(os.path.join(log_base_dir, "session_*"))
            max_session = 0
            for s in existing_sessions:
                try:
                    num = int(os.path.basename(s).split("_")[1])
                    if num > max_session:
                        max_session = num
                except:
                    pass
            session_num = max_session + 1
            session_dir = os.path.join(log_base_dir, f"session_{session_num}")
            os.makedirs(session_dir)
            
            # Khởi tạo file ghi chú thông tin
            self.info_path = os.path.join(session_dir, "session_info.txt")
            start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.info_path, 'w', encoding='utf-8') as f:
                f.write(f"Session {session_num}\n")
                f.write(f"Bắt đầu: {start_time_str}\n")
            
            # Khởi tạo file CSV
            csv_path = os.path.join(session_dir, "speed_track_debug.csv")
            self.csv_file = open(csv_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'state', 'front_dist', 'closest_angle', 'closest_dist', 'current_offset_px', 'steering'])

            # Khởi tạo VideoWriter (ghi ảnh camera và Lidar)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            cam_vid_path = os.path.join(session_dir, "camera_log.mp4")
            lidar_vid_path = os.path.join(session_dir, "lidar_log.mp4")
            
            # Kích thước cố định 300x300, tốc độ 20fps
            self.cam_writer = cv2.VideoWriter(cam_vid_path, fourcc, 20.0, (300, 300))
            self.lidar_writer = cv2.VideoWriter(lidar_vid_path, fourcc, 20.0, (300, 300))

    def log(self, message):
        """In log ra màn hình nếu đang ở chế độ debug."""
        if self.debug_mode:
            print(f"[DEBUG] {message}")
            
    def log_csv(self, state, front_dist, closest_angle, closest_dist, offset_px, steering):
        """Lưu trữ dữ liệu vào CSV sau mỗi chu kỳ."""
        if self.debug_mode and self.csv_writer:
            self.csv_writer.writerow([
                time.time(), state, 
                round(front_dist, 3), 
                round(closest_angle, 2), 
                round(closest_dist, 3), 
                round(offset_px, 1), 
                round(steering, 3)
            ])

    def show_image(self, window_name, image):
        """Hiển thị ảnh bằng OpenCV nếu đang ở chế độ debug."""
        if self.debug_mode:
            try:
                cv2.imshow(window_name, image)
                cv2.waitKey(1)
            except Exception as e:
                print(f"[WARNING] Không thể hiển thị ảnh: {e}")

    def visualize_lidar(self, scan_data):
        """Vẽ dữ liệu Lidar ra một bức ảnh đen 300x300."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cx, cy = 150, 150
        # Vẽ vị trí robot (Tâm màu đỏ)
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
        
        # Tỷ lệ hiển thị: 30 pixels = 1 mét (hiển thị tầm nhìn khoảng 5 mét bán kính)
        scale = 30.0 
        for angle_deg, dist in scan_data:
            # Chuyển đổi góc (0 độ là hướng lên trên) -> trục tung lùi lại 90 độ
            angle_rad = math.radians(angle_deg - 90)
            x = int(cx + dist * scale * math.cos(angle_rad))
            y = int(cy + dist * scale * math.sin(angle_rad))
            
            if 0 <= x < 300 and 0 <= y < 300:
                cv2.circle(img, (x, y), 1, (255, 255, 255), -1)
        
        # Thêm lưới toạ độ (các vòng tròn bán kính 1m, 2m, 3m)
        for r in range(1, 6):
            cv2.circle(img, (cx, cy), int(r * scale), (50, 50, 50), 1)
            
        return img

    def close(self):
        if self.csv_file:
            self.csv_file.close()
        if self.cam_writer:
            self.cam_writer.release()
        if self.lidar_writer:
            self.lidar_writer.release()
            
        if hasattr(self, 'info_path') and self.info_path:
            import datetime
            end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.info_path, 'a', encoding='utf-8') as f:
                f.write(f"Kết thúc: {end_time_str}\n")

    def process(self, blackboard):
        if not self.debug_mode:
            return
            
        state = blackboard.get('state_name', 'UNKNOWN')
        front_dist = blackboard.get('front_dist', 999.0)
        closest_angle = blackboard.get('closest_angle', 0.0)
        closest_dist = blackboard.get('front_dist', 999.0)
        offset_px = blackboard.get('current_offset_px', 0.0)
        steering = blackboard.get('steering', 0.0)
        
        # Ghi log CSV
        self.log_csv(state, front_dist, closest_angle, closest_dist, offset_px, steering)
        
        # Ghi log Video Camera
        latest_image = blackboard.get('latest_image')
        if latest_image is not None and self.cam_writer:
            # Đảm bảo ảnh đúng kích thước trước khi ghi
            img_h, img_w = latest_image.shape[:2]
            if img_h != 300 or img_w != 300:
                resized_img = cv2.resize(latest_image, (300, 300))
                self.cam_writer.write(resized_img)
            else:
                self.cam_writer.write(latest_image)
                
        # Ghi log Video Lidar
        latest_scan = blackboard.get('latest_scan')
        if latest_scan is not None and self.lidar_writer:
            lidar_img = self.visualize_lidar(latest_scan)
            self.lidar_writer.write(lidar_img)
