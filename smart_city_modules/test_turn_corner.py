import os
import sys

# Khắc phục lỗi Illegal instruction (core dumped) phổ biến trên Jetson Nano
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
    print("Cảnh báo: Không tìm thấy rospy.")

from src.core.control.racer_controller import RacerController
from smart_city_modules.autonomous_modules import GoStraightModule, TurnModule
from smart_city_modules.yolo_onnx import YoloONNX

CAMERA_TOPIC = '/csi_cam_0/image_raw'

class TestCornerRunner:
    def __init__(self):
        if HAS_ROS:
            rospy.init_node('test_corner_node', anonymous=True)
            rospy.Subscriber(CAMERA_TOPIC, Image, self.camera_callback)
            
        self.car = RacerController(config={"I2C_ADDRESS": 0x40})
        
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo.onnx'))
        class_names = ['Corner', 'Decision', 'Forbidden', 'Green_Light', 'Interact', 'turn_left', 'Red_Light', 'turn_right', 'straight']
        self.yolo_model = YoloONNX(model_path, class_names)
        
        self.straight_ctrl = GoStraightModule(base_speed=0.6)
        self.turn_ctrl = TurnModule(turn_duration=1.5, max_speed=0.5, max_steering=1.0)
        
        self.latest_frame = None
        
        # Mở file log
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f'test_corner_{int(time.time())}.log')
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
        print("BÀI TEST 2: ĐI TỚI CORNER VÀ THỰC HIỆN RẼ RỒI DỪNG (LOCAL ONNX)")
        print("==================================================")
        
        if not HAS_ROS:
            print("CẢNH BÁO: Không có ROS! Bạn cần ROS để chạy bài test này với gscam.")
            return

        print(f"Đang chờ frame đầu tiên từ topic {CAMERA_TOPIC}...")
        while self.latest_frame is None and not rospy.is_shutdown():
            time.sleep(0.1)
            
        rate = rospy.Rate(20)
        turn_completed = False
        prev_time = time.time()
        
        # Cấu hình VideoWriter
        video_writer = None
        video_path = self.log_path.replace('.log', '.mp4')
        
        try:
            while not turn_completed and not rospy.is_shutdown():
                if self.latest_frame is None:
                    rate.sleep()
                    continue
                    
                frame = self.latest_frame.copy()
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time
                
                h, w = frame.shape[:2]
                self.straight_ctrl.update_resolution(w, h)
                self.turn_ctrl.img_width = w
                
                # Khởi tạo VideoWriter khi có frame đầu tiên
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
                    print(f"Bắt đầu ghi video tại: {video_path}")
                
                # 1. Nhận Detections từ Local ONNX
                detections, annotated_frame = self.yolo_model.infer_and_track(frame, conf_threshold=0.25)
                
                if video_writer is not None:
                    video_writer.write(annotated_frame)
                
                # 2. Tìm Corner node gần nhất
                corners = [d for d in detections if d['label'] == 'Corner']
                target_node = max(corners, key=lambda n: n["y"]) if corners else None
                
                # 3. Logic điều khiển (Bám đường -> Tới gần -> Rẽ)
                if self.turn_ctrl.is_turning:
                    speed, steering = self.turn_ctrl.process()
                    if speed is not None:
                        self.car.steer(steering, speed)
                        msg = f"[FPS: {fps:.1f}] Đang thực thi rẽ {self.turn_ctrl.current_direction}... | Góc lái: {steering:.2f}\n"
                        print(msg, end='')
                        self.log_file.write(msg)
                    else:
                        msg = "\n-> HOÀN THÀNH GÓC RẼ! Kết thúc bài test.\n"
                        print(msg)
                        self.log_file.write(msg)
                        self.car.stop()
                        turn_completed = True
                else:
                    if target_node:
                        distance_estimated = h - target_node['y']
                        if distance_estimated < (h * 0.4):
                            direction = "turn_left" if target_node['x'] < (w / 2) else "turn_right"
                            msg = f"\n=> Tới điểm rẽ (cách {distance_estimated:.0f}px). KÍCH HOẠT RẼ {direction.upper()}!\n"
                            print(msg)
                            self.log_file.write(msg)
                            self.turn_ctrl.start_turn(direction)
                        else:
                            speed, steering = self.straight_ctrl.calculate_command([target_node])
                            self.car.steer(steering, speed)
                            msg = f"[FPS: {fps:.1f}] Tiến tới Corner (cách {distance_estimated:.0f}px) | Góc lái: {steering:.2f}\n"
                            print(msg, end='')
                            self.log_file.write(msg)
                    else:
                        self.car.stop()
                        msg = "Không thấy Corner Node. Đang chờ...\n"
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
            print("Đã tắt camera, đóng log/video và dừng xe an toàn.")

if __name__ == "__main__":
    runner = TestCornerRunner()
    runner.run()
