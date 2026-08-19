#!/usr/bin/env python3
"""
Speed Track - AI SEGMENTATION MODE
Thay thế hoàn toàn bộ nhận diện vạch kẻ đường (HSV, RANSAC) bằng Mô hình AI YOLOv8-Seg.
Ý tưởng: Mô hình AI phân đoạn (segment) vùng mặt đường thành một "Mặt nạ" (Mask).
Xe sẽ tìm trọng tâm của cái Mask này để tính góc lái.
"""
import sys, os, time, math
import cv2, numpy as np
import rospy
from sensor_msgs.msg import Image
from src.core.control.racer_controller import RacerController

# Cố gắng import Ultralytics YOLO. Nếu trên Jetson đã export sang TensorRT thì dùng code TensorRT riêng.
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

class SpeedRacingAI:
    def __init__(self, model_path="best.pt"):
        rospy.init_node('speed_racing_ai', anonymous=True)
        rospy.loginfo("=== KHOI TAO SPEED TRACK AI (SEGMENTATION MODE) ===")
        
        self.W, self.H = 640, 480
        self.racer = RacerController()
        self.racer.stop()
        
        # --- TẢI MÔ HÌNH AI ---
        if HAS_YOLO:
            rospy.loginfo(f"Đang tải mô hình YOLO từ: {model_path}...")
            self.model = YOLO(model_path)
        else:
            rospy.logerr("Lỗi: Chưa cài thư viện ultralytics. (pip install ultralytics)")
            sys.exit(1)
            
        # --- ROS ---
        self.latest_image = None
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self._cam_cb, queue_size=1)
        
        self.target_speed = 0.35
        self.loop_rate = rospy.Rate(20) # 20 FPS

    def _cam_cb(self, msg):
        try:
            if 'compressed' in msg.encoding:
                self.latest_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
        except Exception as e:
            pass

    def process_ai(self, frame):
        """Đưa ảnh qua model AI và tính toán góc lái dựa trên Vùng Xanh (Drivable Area)"""
        # 1. Chạy AI Inference
        # imgsz=320 để model chạy nhẹ hơn trên GPU của Jetson
        results = self.model.predict(source=frame, imgsz=320, verbose=False, conf=0.5)
        
        result = results[0]
        dbg_image = result.plot() # Lấy ảnh đã vẽ sẵn mask của YOLO để debug
        
        steer_angle = 0.0
        
        # 2. Rút trích Mask (Mặt nạ vùng đường)
        if result.masks is not None:
            # Lấy mask đầu tiên (giả sử chỉ có 1 class là 'drivable_area')
            mask = result.masks.data[0].cpu().numpy()
            
            # 3. Thuật toán tìm Tâm đường (Midpoint) bằng lát cắt ngang
            # Khắc phục triệt để lỗi sai số tích lũy khi bị khuất mép đường
            y_lookahead = int(self.H * 0.65) # Lát cắt ở mức 65% màn hình
            row = mask[y_lookahead, :]
            road_pixels = np.where(row > 0)[0]
            
            if len(road_pixels) > 0:
                x_left = road_pixels[0]
                x_right = road_pixels[-1]
                
                cX = int((x_left + x_right) / 2)
                cY = y_lookahead
                
                # Bù trừ sai số khi camera bị khuất một bên
                if x_left <= 5 and x_right < self.W - 10:
                    cX = cX - int(self.W * 0.15) # Dịch tâm sang trái
                elif x_right >= self.W - 5 and x_left > 10:
                    cX = cX + int(self.W * 0.15) # Dịch tâm sang phải
                
                # Vẽ điểm trọng tâm lên ảnh debug
                cv2.circle(dbg_image, (cX, cY), 10, (255, 0, 255), -1)
                
                # 4. Tính toán góc lái (Steering)
                # Điểm trung tâm của ảnh là self.W / 2
                error_x = cX - (self.W / 2)
                
                # Hàm P-Controller cơ bản: Steer = Kp * error
                Kp = 0.005
                steer_angle = error_x * Kp
                steer_angle = max(-1.0, min(1.0, steer_angle)) # Cắt ở khoảng [-1, 1]
                
                # Vẽ tia chỉ đạo
                cv2.line(dbg_image, (int(self.W/2), self.H), (cX, cY), (0, 255, 255), 3)

        return steer_angle, dbg_image

    def run(self):
        rospy.loginfo("Đang đợi ảnh từ camera...")
        while self.latest_image is None and not rospy.is_shutdown():
            time.sleep(0.1)
            
        rospy.loginfo("=== XE BẮT ĐẦU CHẠY BẰNG AI ===")
        
        while not rospy.is_shutdown():
            frame = self.latest_image.copy()
            
            # Xử lý AI
            steer, dbg = self.process_ai(frame)
            
            # Điều khiển xe
            self.racer.steer(steer)
            self.racer.throttle(self.target_speed)
            
            # Hiển thị
            cv2.imshow("AI Segmentation Mode", dbg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            self.loop_rate.sleep()
            
        self.racer.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Có thể truyền đường dẫn model qua argument hoặc fix cứng
    model_file = "runs/segment/jetson_track_seg-2/weights/best.pt" 
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
        
    ai_racer = SpeedRacingAI(model_path=model_file)
    ai_racer.run()
