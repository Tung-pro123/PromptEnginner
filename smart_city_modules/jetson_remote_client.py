import socket
import cv2
import time
import numpy as np
import struct
import json
import sys
import os

# Thêm đường dẫn để import thư viện điều khiển xe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smart_city_modules.control_modules import CarController, HAS_ROS
from smart_city_modules.autonomous_modules import GoStraightModule, TurnModule, DecisionModule

if HAS_ROS:
    import rospy
    from sensor_msgs.msg import Image

# ========================================================
# CẤU HÌNH IP CỦA LAPTOP (SERVER) TRONG MẠNG LAN CỦA BẠN
SERVER_IP = '192.168.1.X' 
PORT = 5000
# Tên topic camera của gscam (Sửa nếu cần thiết)
CAMERA_TOPIC = '/camera/image_raw' 
# ========================================================

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: 
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

class JetsonRemoteClient:
    def __init__(self):
        if HAS_ROS:
            rospy.init_node('remote_yolo_client_node', anonymous=True)
            
        self.car = CarController()
        
        # Khởi tạo các module điều khiển tự lái (Mặc định 640x480, sẽ update sau khi có frame)
        self.straight_ctrl = GoStraightModule(img_width=640, img_height=480, base_speed=0.3)
        self.turn_ctrl = TurnModule(img_width=640, turn_duration=1.5, max_speed=0.4, max_steering=1.0)
        
        self.latest_frame = None
        
        # Kết nối TCP tới Laptop
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Đang kết nối tới Laptop tại {SERVER_IP}:{PORT}...")
        try:
            self.client_socket.connect((SERVER_IP, PORT))
            print("Đã kết nối tới Laptop thành công!")
        except Exception as e:
            print(f"Không thể kết nối tới máy chủ. Vui lòng kiểm tra IP và tường lửa. Lỗi: {e}")
            sys.exit(1)
            
        # Đăng ký ROS Subscriber để lấy ảnh từ gscam
        if HAS_ROS:
            rospy.Subscriber(CAMERA_TOPIC, Image, self.camera_callback)
            print(f"Đã đăng ký nhận video từ ROS topic: {CAMERA_TOPIC}")
        else:
            print("Không tìm thấy ROS! Vui lòng chạy trên môi trường có ROS để nhận gscam.")
            sys.exit(1)

    def camera_callback(self, msg):
        try:
            if 'compressed' in msg.encoding:
                np_arr = np.frombuffer(msg.data, np.uint8)
                self.latest_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                self.latest_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
                
            # Cập nhật resolution
            h, w = self.latest_frame.shape[:2]
            self.straight_ctrl.update_resolution(w, h)
            self.turn_ctrl.img_width = w
        except Exception as e:
            print(f"Lỗi đọc camera ROS: {e}")

    def run_loop(self):
        frame_count = 0
        prev_time = time.time()
        
        print("Đang chờ nhận frame đầu tiên từ gscam...")
        while self.latest_frame is None and not rospy.is_shutdown():
            time.sleep(0.1)
            
        print("Đã có hình ảnh, bắt đầu vòng lặp điều khiển xe!")
        rate = rospy.Rate(20)
        
        try:
            while not rospy.is_shutdown():
                if self.latest_frame is None:
                    rate.sleep()
                    continue
                
                frame = self.latest_frame.copy()
                
                # Tính FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time
                frame_count += 1
                
                # 1. Mã hoá ảnh JPEG (Giảm dung lượng đường truyền)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                result, img_encoded = cv2.imencode('.jpg', frame, encode_param)
                data_bytes = img_encoded.tobytes()
                
                # Gửi ảnh lên Laptop
                self.client_socket.sendall(struct.pack('<I', len(data_bytes)))
                self.client_socket.sendall(data_bytes)
                
                # 2. Nhận kết quả từ Laptop
                lengthbuf = recvall(self.client_socket, 4)
                if not lengthbuf: break
                length, = struct.unpack('<I', lengthbuf)
                
                json_bytes = recvall(self.client_socket, length)
                if not json_bytes: break
                
                detections = json.loads(json_bytes.decode('utf-8'))
                
                # Vẽ log text đơn giản lên cửa sổ Jetson
                for d in detections:
                    label_text = f"{d['label']} {d.get('id', '')}"
                    cv2.putText(frame, label_text, (int(d['x']), int(d['y'])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imshow("Jetson Remote View", frame)
                cv2.waitKey(1)
                
                # 3. Chạy Logic Rẽ / Đi thẳng
                self.turn_ctrl.trigger_turn_if_needed(detections)
                turn_speed, turn_steering = self.turn_ctrl.process()
                
                if turn_speed is not None and turn_steering is not None:
                    # Xe đang trong pha rẽ
                    self.car.steer(turn_steering, turn_speed)
                    if frame_count % 10 == 0:
                        print(f"[FPS: {fps:.1f}] Đang rẽ {self.turn_ctrl.current_direction}...")
                else:
                    # Xe đi thẳng
                    straight_speed, straight_steering = self.straight_ctrl.calculate_command(detections)
                    if straight_speed is not None and straight_steering is not None:
                        self.car.steer(straight_steering, straight_speed)
                        if frame_count % 10 == 0:
                            print(f"[FPS: {fps:.1f}] Đi thẳng | Góc lái: {straight_steering:.2f}")
                    else:
                        if frame_count % 10 == 0:
                            print(f"[FPS: {fps:.1f}] Đang dò đường...")
                
                rate.sleep()
                    
        except KeyboardInterrupt:
            print("\nĐã dừng xe thủ công.")
        except Exception as e:
            print(f"\nLỗi trong quá trình chạy: {e}")
        finally:
            self.car.stop()
            self.client_socket.close()
            cv2.destroyAllWindows()
            print("Hoàn tất dọn dẹp hệ thống.")

if __name__ == '__main__':
    client = JetsonRemoteClient()
    client.run_loop()
