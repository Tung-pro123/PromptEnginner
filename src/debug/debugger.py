import csv
import time
import os

class Debugger:
    """Lớp tiện ích để hỗ trợ debug chương trình."""

    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.csv_file = None
        self.csv_writer = None
        
        if self.debug_mode:
            # Khởi tạo file CSV
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed_track_debug.csv")
            self.csv_file = open(log_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'state', 'front_dist', 'closest_angle', 'closest_dist', 'current_offset_px', 'steering'])

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
                import cv2
                cv2.imshow(window_name, image)
                cv2.waitKey(1)
            except ImportError:
                print("[WARNING] OpenCV (cv2) chưa được cài đặt để hiển thị ảnh.")
                
    def close(self):
        if self.csv_file:
            self.csv_file.close()
