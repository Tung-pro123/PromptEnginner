import cv2
import time
import numpy as np
import sys
import os

# Thêm đường dẫn thư mục gốc để có thể import từ smart_city_modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_city_modules.control_modules import CarController, HAS_ROS
from smart_city_modules.autonomous_modules import GoStraightModule, TurnModule
from smart_city_modules.remote_yolo_client import RemoteYOLOClient

if HAS_ROS:
    import rospy
    from sensor_msgs.msg import Image

# Thay đổi IP này thành IP của Laptop
SERVER_IP = '192.168.1.X'
CAMERA_TOPIC = '/camera/image_raw'

class TestCornerRunner:
    def __init__(self):
        if HAS_ROS:
            rospy.init_node('test_corner_node', anonymous=True)
            rospy.Subscriber(CAMERA_TOPIC, Image, self.camera_callback)
            
        self.car = CarController()
        self.yolo_client = RemoteYOLOClient(server_ip=SERVER_IP, port=5000)
        
        self.straight_ctrl = GoStraightModule(base_speed=0.3)
        self.turn_ctrl = TurnModule(turn_speed=0.3, max_steering=1.0, turn_duration=1.5)
        
        self.latest_frame = None

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
        print("BÀI TEST 2: ĐI TỚI CORNER VÀ THỰC HIỆN RẼ RỒI DỪNG (REMOTE YOLO + ROS)")
        print("==================================================")
        
        if not HAS_ROS:
            print("CẢNH BÁO: Không có ROS! Bạn cần ROS để chạy bài test này với gscam.")
            return

        print(f"Đang chờ frame đầu tiên từ topic {CAMERA_TOPIC}...")
        while self.latest_frame is None and not rospy.is_shutdown():
            time.sleep(0.1)
            
        rate = rospy.Rate(20)
        turn_completed = False
        
        try:
            while not turn_completed and not rospy.is_shutdown():
                if self.latest_frame is None:
                    rate.sleep()
                    continue
                    
                frame = self.latest_frame.copy()
                h, w = frame.shape[:2]
                self.straight_ctrl.update_resolution(w, h)
                self.turn_ctrl.img_width = w
                
                # 1. Nhận Detections từ Laptop
                detections = self.yolo_client.get_detections(frame)
                
                # 2. Tìm Corner node gần nhất
                corners = [d for d in detections if d['label'] == 'Corner']
                target_node = max(corners, key=lambda n: n["y"]) if corners else None
                
                # Hiển thị tracking
                annotated_frame = self.yolo_client.draw_detections(frame, detections)
                cv2.imshow("Test Corner Node (Remote + ROS)", annotated_frame)
                cv2.waitKey(1)
                
                # 3. Logic điều khiển (Bám đường -> Tới gần -> Rẽ)
                if self.turn_ctrl.is_turning:
                    speed, steering = self.turn_ctrl.process()
                    if speed is not None:
                        self.car.steer(steering, speed)
                        print(f"Đang thực thi rẽ {self.turn_ctrl.current_direction}... | Góc lái: {steering:.2f}")
                    else:
                        print("\n-> HOÀN THÀNH GÓC RẼ! Kết thúc bài test.")
                        self.car.stop()
                        turn_completed = True
                else:
                    if target_node:
                        distance_estimated = h - target_node['y']
                        if distance_estimated < (h * 0.4):
                            direction = "turn_left" if target_node['x'] < (w / 2) else "turn_right"
                            print(f"\n=> Tới điểm rẽ (cách {distance_estimated:.0f}px). KÍCH HOẠT RẼ {direction.upper()}!")
                            self.turn_ctrl.start_turn(direction)
                        else:
                            speed, steering = self.straight_ctrl.calculate_command([target_node])
                            self.car.steer(steering, speed)
                            print(f"Tiến tới Corner (cách {distance_estimated:.0f}px) | Góc lái: {steering:.2f}")
                    else:
                        self.car.stop()
                        print("Không thấy Corner Node. Đang chờ...")
                
                rate.sleep()
                
        except KeyboardInterrupt:
            print("\nDừng thủ công bằng Ctrl+C.")
        finally:
            self.car.stop()
            self.yolo_client.close()
            cv2.destroyAllWindows()
            print("Đã tắt camera và dừng xe an toàn.")

if __name__ == "__main__":
    runner = TestCornerRunner()
    runner.run()
