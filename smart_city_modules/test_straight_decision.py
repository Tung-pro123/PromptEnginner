import cv2
import time
import numpy as np
import sys
import os

# Thêm đường dẫn thư mục gốc để có thể import từ smart_city_modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_city_modules.control_modules import CarController, HAS_ROS
from smart_city_modules.autonomous_modules import GoStraightModule
from smart_city_modules.remote_yolo_client import RemoteYOLOClient

if HAS_ROS:
    import rospy
    from sensor_msgs.msg import Image

# Thay đổi IP này thành IP của Laptop
SERVER_IP = '192.168.1.X'
CAMERA_TOPIC = '/camera/image_raw'

class TestDecisionRunner:
    def __init__(self):
        if HAS_ROS:
            rospy.init_node('test_decision_node', anonymous=True)
            rospy.Subscriber(CAMERA_TOPIC, Image, self.camera_callback)
            
        self.car = CarController()
        self.yolo_client = RemoteYOLOClient(server_ip=SERVER_IP, port=5000)
        self.straight_ctrl = GoStraightModule(base_speed=0.3)
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
        print("BÀI TEST 1: ĐI THẲNG TỚI DECISION NODE RỒI DỪNG (REMOTE YOLO + ROS)")
        print("==================================================")
        
        if not HAS_ROS:
            print("CẢNH BÁO: Không có ROS! Bạn cần ROS để chạy bài test này với gscam.")
            return

        print(f"Đang chờ frame đầu tiên từ topic {CAMERA_TOPIC}...")
        while self.latest_frame is None and not rospy.is_shutdown():
            time.sleep(0.1)
            
        rate = rospy.Rate(20)
        
        try:
            while not rospy.is_shutdown():
                if self.latest_frame is None:
                    rate.sleep()
                    continue
                    
                frame = self.latest_frame.copy()
                h, w = frame.shape[:2]
                self.straight_ctrl.update_resolution(w, h)
                
                # 1. Nhận Detections từ Laptop
                detections = self.yolo_client.get_detections(frame)
                
                # 2. Tìm Decision node gần nhất (dựa trên y)
                decisions = [d for d in detections if d['label'] == 'Decision']
                target_node = max(decisions, key=lambda n: n["y"]) if decisions else None
                
                # Hiển thị tracking
                annotated_frame = self.yolo_client.draw_detections(frame, detections)
                cv2.imshow("Test Decision Node (Remote + ROS)", annotated_frame)
                cv2.waitKey(1)
                
                # 3. Điều khiển xe
                if target_node:
                    speed, steering = self.straight_ctrl.calculate_command([target_node])
                    self.car.steer(steering, speed)
                    
                    distance_to_bottom = h - target_node['y']
                    print(f"Đang bám Decision (y={target_node['y']:.0f}, cách đáy {distance_to_bottom:.0f}px) | Góc lái: {steering:.2f}")
                    
                    if target_node['y'] > h * 0.8:
                        print("\n-> ĐÃ TỚI DECISION NODE! Kết thúc bài test.")
                        self.car.stop()
                        break
                else:
                    self.car.stop()
                    print("Không thấy Decision Node. Đang chờ...")
                    
                rate.sleep()
                    
        except KeyboardInterrupt:
            print("\nDừng thủ công bằng Ctrl+C.")
        finally:
            self.car.stop()
            self.yolo_client.close()
            cv2.destroyAllWindows()
            print("Đã dừng xe an toàn.")

if __name__ == "__main__":
    runner = TestDecisionRunner()
    runner.run()
