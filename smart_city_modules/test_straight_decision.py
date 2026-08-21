import os
import sys

# Khắc phục lỗi Illegal instruction (core dumped) phổ biến trên Jetson Nano 
# do thư viện numpy/OpenBLAS không nhận diện đúng CPU kiến trúc cũ (ARMv8.0/A57)
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"

import cv2
import time
import numpy as np

# Thêm đường dẫn thư mục gốc để có thể import từ smart_city_modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import rospy
    from sensor_msgs.msg import Image
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

from src.core.control.racer_controller import RacerController
from smart_city_modules.autonomous_modules import GoStraightModule
from smart_city_modules.yolo_onnx import YoloONNX

CAMERA_TOPIC = '/csi_cam_0/image_raw'

class TestDecisionRunner:
    def __init__(self):
        if HAS_ROS:
            rospy.init_node('test_decision_node', anonymous=True)
            rospy.Subscriber(CAMERA_TOPIC, Image, self.camera_callback)
            
        self.car = RacerController(config={"I2C_ADDRESS": 0x40})
        
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo.onnx'))
        class_names = ['Corner', 'Decision', 'Forbidden', 'Green_Light', 'Interact', 'turn_left', 'Red_Light', 'turn_right', 'straight']
        self.yolo_model = YoloONNX(model_path, class_names)
        
        self.straight_ctrl = GoStraightModule(base_speed=0.6)
        self.latest_frame = None
        
        # Mở file log
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f'test_straight_{int(time.time())}.log')
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        print(f"File log được lưu tại: {self.log_path}")

    def camera_callback(self, msg):
        try:
            if 'compressed' in msg.encoding:
                np_arr = np.frombuffer(msg.data, np.uint8)
                self.latest_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                self.latest_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
        except Exception as e:
            print(f"Lỗi camera ROS: {e}")

    def run(self):
        print("==================================================")
        print("BÀI TEST 1: ĐI THẲNG TỚI DECISION NODE RỒI DỪNG (LOCAL ONNX)")
        print("==================================================")
        
        if not HAS_ROS:
            print("CẢNH BÁO: Không có ROS! Bạn cần ROS để chạy bài test này với gscam.")
            return

        print(f"Đang chờ frame đầu tiên từ topic {CAMERA_TOPIC}...")
        while self.latest_frame is None and not rospy.is_shutdown():
            time.sleep(0.1)
            
        rate = rospy.Rate(20)
        prev_time = time.time()
        
        # Cấu hình VideoWriter
        video_writer = None
        video_path = self.log_path.replace('.log', '.mp4')
        
        try:
            while not rospy.is_shutdown():
                if self.latest_frame is None:
                    rate.sleep()
                    continue
                    
                frame = self.latest_frame.copy()
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time
                h, w = frame.shape[:2]
                self.straight_ctrl.update_resolution(w, h)
                
                # Khởi tạo VideoWriter khi có frame đầu tiên
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
                    print(f"Bắt đầu ghi video tại: {video_path}")
                
                # 1. Nhận Detections từ Local ONNX
                detections, annotated_frame = self.yolo_model.infer_and_track(frame, conf_threshold=0.25)
                
                if video_writer is not None:
                    video_writer.write(annotated_frame)
                
                # 2. Tìm Decision node gần nhất (dựa trên y)
                decisions = [d for d in detections if d['label'] == 'Decision']
                target_node = max(decisions, key=lambda n: n["y"]) if decisions else None
                
                # 3. Điều khiển xe
                if target_node:
                    speed, steering = self.straight_ctrl.calculate_command([target_node])
                    self.car.steer(steering, speed)
                    
                    distance_to_bottom = h - target_node['y']
                    log_msg = f"[FPS: {fps:.1f}] Đang bám Decision (y={target_node['y']:.0f}, cách đáy {distance_to_bottom:.0f}px) | Góc lái: {steering:.2f}\n"
                    print(log_msg, end='')
                    self.log_file.write(log_msg)
                    
                    if target_node['y'] > h * 0.8:
                        msg = "\n-> ĐÃ TỚI DECISION NODE! Kết thúc bài test.\n"
                        print(msg)
                        self.log_file.write(msg)
                        self.car.stop()
                        break
                else:
                    self.car.stop()
                    msg = "Không thấy Decision Node. Đang chờ...\n"
                    print(msg, end='')
                    self.log_file.write(msg)
                    
                rate.sleep()
                    
        except KeyboardInterrupt:
            msg = "\nDừng thủ công bằng Ctrl+C.\n"
            print(msg)
            self.log_file.write(msg)
        finally:
            self.car.stop()
            self.log_file.close()
            if video_writer is not None:
                video_writer.release()
            print("Đã dừng xe an toàn và đóng file log, video.")

if __name__ == "__main__":
    runner = TestDecisionRunner()
    runner.run()
