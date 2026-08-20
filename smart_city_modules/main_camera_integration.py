#!/usr/bin/env python3
import sys
import os
import time
import cv2
import numpy as np

# Thêm đường dẫn thư mục gốc để import module xe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import ROS và các thư viện cần thiết
try:
    import rospy
    from sensor_msgs.msg import Image
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("Cảnh báo: Không tìm thấy rospy. Code sẽ chạy nhưng không nhận được camera từ ROS.")

# Import module tự hành vừa code và controller xe
from smart_city_modules.autonomous_modules import GoStraightModule, TurnModule
from src.core.control.racer_controller import RacerController

class SmartCityCameraRunner:
    def __init__(self):
        print("=== KHỞI TẠO HỆ THỐNG SMART CITY CAMERA ===")
        
        # Khởi tạo phần cứng xe
        self.car = RacerController()
        self.car.stop()
        
        # Khởi tạo 2 module thuật toán độc lập
        self.go_straight_ctrl = GoStraightModule(img_width=640, img_height=480, base_speed=0.3)
        self.turn_ctrl = TurnModule(img_width=640, turn_duration=2.5, max_speed=0.4, max_steering=1.0)
        
        # Khởi tạo biến lưu trữ frame camera
        self.latest_frame = None
        
        # Nếu có ROS, đăng ký lắng nghe topic camera
        if HAS_ROS:
            rospy.init_node('smart_city_yolo_node', anonymous=True)
            rospy.Subscriber('/csi_cam_0/image_raw', Image, self._camera_callback)
            print("Đã đăng ký Subscriber camera: /csi_cam_0/image_raw")
        
    def _camera_callback(self, msg):
        """Callback được gọi mỗi khi ROS gửi một frame ảnh mới từ camera"""
        try:
            if 'compressed' in msg.encoding:
                np_arr = np.frombuffer(msg.data, np.uint8)
                self.latest_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                self.latest_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
                
            # Cập nhật resolution nếu frame khác mặc định
            h, w = self.latest_frame.shape[:2]
            self.go_straight_ctrl.update_resolution(w, h)
            self.turn_ctrl.img_width = w
        except Exception as e:
            print(f"Lỗi đọc camera: {e}")

    def run_yolo_inference(self, frame):
        """
        [MÔ PHỎNG] Hàm này là nơi bạn đặt code chạy model YOLO-Segmentation.
        Trả về danh sách dictionary theo format của module.
        """
        # =====================================================================
        # TODO: BẠN THAY ĐOẠN CODE NÀY BẰNG CODE INFERENCE CỦA BẠN (ultralytics)
        # Ví dụ:
        # results = model.predict(frame)
        # detections = []
        # for box, mask, cls in zip(results[0].boxes, results[0].masks, results[0].boxes.cls):
        #     label = model.names[int(cls)]
        #     x, y = get_center_from_mask(mask)
        #     detections.append({"label": label, "x": x, "y": y})
        # return detections
        # =====================================================================
        
        # Tạm thời trả về danh sách rỗng để code không bị lỗi
        return []

    def run_loop(self):
        """Vòng lặp chính xử lý ảnh và điều khiển xe"""
        print("Bắt đầu vòng lặp điều khiển. Nhấn Ctrl+C để dừng.")
        
        # Cho ROS rate khoảng 20Hz (20 fps)
        rate = rospy.Rate(20) if HAS_ROS else None
        
        while (not HAS_ROS) or (not rospy.is_shutdown()):
            if self.latest_frame is None:
                if HAS_ROS: rate.sleep()
                else: time.sleep(0.05)
                continue
            
            frame_to_process = self.latest_frame.copy()
            
            # 1. Chạy AI lấy kết quả
            detections = self.run_yolo_inference(frame_to_process)
            
            # 2. Xử lý thuật toán rẽ trước (Ưu tiên)
            # Kiểm tra xem có cần trigger rẽ (ngã tư, góc cua)
            self.turn_ctrl.trigger_turn_if_needed(detections)
            
            # Tiến hành rẽ (nếu đang trong trạng thái rẽ)
            turn_speed, turn_steering = self.turn_ctrl.process()
            
            if turn_speed is not None and turn_steering is not None:
                # Đang rẽ -> Áp dụng lệnh rẽ
                self.car.steer(turn_steering, turn_speed)
            else:
                # 3. Nếu không rẽ, xử lý đi thẳng
                straight_speed, straight_steering = self.go_straight_ctrl.calculate_command(detections)
                
                if straight_speed is not None and straight_steering is not None:
                    self.car.steer(straight_steering, straight_speed)
                else:
                    # Tạm dừng nếu không thấy đường hoặc không thấy nhận diện
                    # self.car.stop()
                    pass
            
            # Chờ frame tiếp theo
            if HAS_ROS:
                rate.sleep()
            else:
                time.sleep(0.05)
                
        # Khi kết thúc
        self.car.stop()

if __name__ == "__main__":
    try:
        runner = SmartCityCameraRunner()
        # Đợi 2s cho camera khởi động và nhận frame
        time.sleep(2)
        runner.run_loop()
    except KeyboardInterrupt:
        print("\nĐã dừng thủ công.")
    finally:
        runner.car.stop()
        print("Đã dừng xe an toàn.")
