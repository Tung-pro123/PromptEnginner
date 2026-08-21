#!/usr/bin/env python3
import os
import sys

# Khắc phục lỗi Illegal instruction (core dumped) phổ biến trên Jetson Nano
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"

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
        self.car = RacerController(config={"I2C_ADDRESS": 0x40})
        self.car.stop()
        
        # Khởi tạo 2 module thuật toán độc lập (Đã tăng tốc độ)
        self.go_straight_ctrl = GoStraightModule(img_width=640, img_height=480, base_speed=0.6)
        self.turn_ctrl = TurnModule(img_width=640, turn_duration=1.5, max_speed=0.5, max_steering=1.0)
        
        # Khởi tạo biến lưu trữ frame camera
        self.latest_frame = None

        # Load mô hình YOLO ONNX
        try:
            from smart_city_modules.yolo_onnx import YoloONNX
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo.onnx'))
            
            # Khai báo mảng tên class tương ứng với model huấn luyện của bạn
            class_names = ['Corner', 'Decision', 'Forbidden', 'Green_Light', 'Interact', 'turn_left', 'Red_Light', 'turn_right', 'straight']
            
            self.model = YoloONNX(model_path, class_names)
            self.has_model = True
            print(f"Đã load mô hình ONNX từ: {model_path}")
        except ImportError:
            self.has_model = False
            print("Cảnh báo: Không tìm thấy thư viện. Bỏ qua nhận diện YOLO.")
        except Exception as e:
            self.has_model = False
            print(f"Lỗi khi load mô hình YOLO ONNX: {e}")
        
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
        Chạy inference mô hình ONNX và trả về kết quả theo format dict cho module xe kèm theo frame đã vẽ label.
        """
        if not hasattr(self, 'has_model') or not self.has_model:
            return [], frame.copy()

        # Gọi module YoloONNX tự code
        detections, annotated_frame = self.model.infer_and_track(frame, conf_threshold=0.25)
        
        return detections, annotated_frame

    def run_loop(self):
        """Vòng lặp chính xử lý ảnh và điều khiển xe"""
        print("Bắt đầu vòng lặp điều khiển. Nhấn Ctrl+C để dừng.")
        
        # Cho ROS rate khoảng 20Hz (20 fps)
        rate = rospy.Rate(20) if HAS_ROS else None
        
        # Cấu hình lưu video log
        video_writer = None
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        os.makedirs(log_dir, exist_ok=True)
        video_path = os.path.join(log_dir, f'run_log_{int(time.time())}.mp4')
        
        # Biến đếm FPS và log
        prev_time = time.time()
        frame_count = 0
        
        while (not HAS_ROS) or (not rospy.is_shutdown()):
            if self.latest_frame is None:
                if HAS_ROS: rate.sleep()
                else: time.sleep(0.05)
                continue
            
            frame_to_process = self.latest_frame.copy()
            
            # Tính FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time + 1e-6)
            prev_time = current_time
            frame_count += 1
            
            # Khởi tạo VideoWriter khi có frame đầu tiên để biết kích thước
            if video_writer is None:
                h, w = frame_to_process.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
                print(f"Bắt đầu ghi log video tại: {video_path}")
            
            # 1. Chạy AI lấy kết quả
            detections, annotated_frame = self.run_yolo_inference(frame_to_process)
            
            # Ghi frame vào video log
            if video_writer is not None:
                video_writer.write(annotated_frame)
            
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
            
            # Log ra màn hình mỗi 10 frame (tránh trôi log)
            if frame_count % 10 == 0:
                labels = [d["label"] for d in detections]
                signs_count = len([l for l in labels if l in ["turn_left", "turn_right", "Forbidden"]])
                nodes_count = len([l for l in labels if l in ["Decision", "Interact", "Corner"]])
                
                status_msg = f"[LOG] FPS: {fps:.1f} | Nodes: {nodes_count} | Biển báo: {signs_count} | Labels: {labels}"
                
                if self.turn_ctrl.is_turning:
                    status_msg += f" | Status: Đang rẽ {self.turn_ctrl.current_direction}"
                else:
                    status_msg += " | Status: Đi thẳng"
                    
                print(status_msg)
            
            # Chờ frame tiếp theo
            if HAS_ROS:
                rate.sleep()
            else:
                time.sleep(0.05)
                
        # Khi kết thúc
        if video_writer is not None:
            video_writer.release()
            print(f"Đã lưu log video tại: {video_path}")
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
