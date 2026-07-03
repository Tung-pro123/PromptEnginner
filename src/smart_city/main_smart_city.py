#!/usr/bin/env python3
"""
Jetson AI Racer Challenge 2026 - Smart City (Bài 2)
Hệ thống điều khiển hoàn toàn tự động không sử dụng bản đồ (Mapless Autonomous).
- Sử dụng thuật toán bám tâm đường bằng cách phát hiện 2 vạch biên trắng (Camera).
- Khi phát hiện giao lộ (vạch kẻ đường bị chia cắt hoặc phát hiện biển báo):
  1. Xe dừng lại trước giao lộ.
  2. Đọc biển báo giao thông (Roboflow API) để quyết định hành động rẽ trái, phải hoặc đi thẳng.
  3. Thực thi đánh lái rẽ 90 độ và tìm lại làn mới để tiếp tục hành trình.
- Gửi kết quả nhận diện biển báo lên Server và kết nối qua MQTT.

Chạy trên xe:
    python3 src/smart_city/main_smart_city.py
"""

import sys
# Sắp xếp lại sys.path để ưu tiên các thư viện Python 3 trước, tránh xung đột với ROS Python 2.7
py3_paths = [p for p in sys.path if 'python2.7' not in p]
py2_paths = [p for p in sys.path if 'python2.7' in p]
sys.path = py3_paths + py2_paths

import os
import rospy
import cv2
import numpy as np
import time
import math
import requests
from enum import Enum
from datetime import datetime, timezone
from sensor_msgs.msg import LaserScan, Image
import paho.mqtt.client as mqtt

# Import các module điều khiển vật lý
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.control.racer_controller import RacerController

class RobotState(Enum):
    STATE_LINE_FOLLOWING = 1     # Đang bám làn bình thường
    STATE_APPROACHING = 2        # Đang tiến vào giao lộ
    STATE_HANDLING_SIGN = 3      # Dừng lại quét biển báo (Roboflow) và quyết định hướng đi
    STATE_TURNING = 4            # Đang thực thi lệnh cua 90 độ
    STATE_REACQUIRING = 5        # Tìm lại làn mới sau khi rẽ
    STATE_GOAL_REACHED = 6       # Đã về đích (Gặp biển Stop hoặc trạm đích)
    STATE_DEAD_END = 7           # Gặp lỗi không phục hồi

class SmartCityController:
    def __init__(self):
        rospy.init_node('smart_city_node', anonymous=True)
        rospy.loginfo("=== KHỞI TẠO BỘ ĐIỀU KHIỂN SMART CITY (KHÔNG DÙNG MAP) ===")
        self.setup_parameters()
        self.initialize_hardware()
        self.initialize_mqtt()

        # Khởi tạo trạng thái
        self.state = RobotState.STATE_LINE_FOLLOWING
        self.latest_scan = None
        self.latest_image = None
        self.state_change_time = rospy.get_time()

        # Video Recorder để debug
        self.video_writer = None
        self.initialize_video_writer()

        # Đăng ký ROS Topics
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)
        rospy.loginfo("Đăng ký nhận dữ liệu từ LiDAR và Camera thành công.")

    def setup_parameters(self):
        """Cấu hình các tham số hoạt động cho sa bàn thành phố thông minh"""
        self.BASE_SPEED = 0.16         # Tốc độ ga cơ bản của Smart City (chạy cẩn thận)
        self.WIDTH = 300
        self.HEIGHT = 300

        # Tham số thời gian chuyển trạng thái giao lộ
        self.INTERSECTION_APPROACH_DURATION = 0.6  # Thời gian bò thêm vào giữa giao lộ (giây)
        self.LINE_REACQUIRE_TIMEOUT = 3.0          # Quá thời gian này không tìm thấy vạch -> Dừng xe

        # Tọa độ quét vạch Camera (Nâng cao lên để tránh quét đè lên cản trước màu xanh lá của xe)
        self.ROI_Y = int(self.HEIGHT * 0.73)
        self.ROI_H = int(self.HEIGHT * 0.15)
        self.LOOKAHEAD_ROI_Y = int(self.HEIGHT * 0.50)
        self.LOOKAHEAD_ROI_H = int(self.HEIGHT * 0.15)

        # File video debug
        self.VIDEO_OUTPUT_FILENAME = 'smart_city_run.avi'
        self.VIDEO_FPS = 20
        self.VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')

        # Cấu hình kết nối Roboflow API để quét biển báo
        self.RF_MODEL = "dataset3-c4kyj"
        self.RF_VERSION = "1"
        self.RF_API_KEY = os.environ.get('ROBOFLOW_API_KEY', '')  # Set biến môi trường ROBOFLOW_API_KEY
        
        # Địa chỉ URL gửi báo cáo biển báo về server cuộc thi
        self.SUBMIT_URL = os.environ.get('SUBMIT_URL', 'http://localhost:8080/api/submit')
        self.TEAM_NAME = os.environ.get('TEAM_NAME', 'PromptEngineer')

        # Cấu hình MQTT
        self.MQTT_BROKER = "localhost"
        self.MQTT_PORT = 1883
        self.MQTT_DATA_TOPIC = "jetbot/corrected_event_data"

    def initialize_hardware(self):
        """Khởi tạo điều khiển phần cứng Ackermann."""
        self.racer = RacerController()
        self.racer.stop()

    def initialize_mqtt(self):
        """Kết nối Broker MQTT gửi dữ liệu hành trình."""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.connect(self.MQTT_BROKER, self.MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            rospy.loginfo("Kết nối Broker MQTT thành công.")
        except Exception as e:
            rospy.logwarn(f"Không thể kết nối MQTT: {e}")
            self.mqtt_client = None

    def initialize_video_writer(self):
        """Khởi tạo ghi video debug."""
        try:
            self.video_writer = cv2.VideoWriter(
                self.VIDEO_OUTPUT_FILENAME,
                self.VIDEO_FOURCC,
                self.VIDEO_FPS,
                (self.WIDTH, self.HEIGHT)
            )
        except Exception as e:
            rospy.logerr(f"Lỗi khởi tạo VideoWriter: {e}")

    def lidar_callback(self, msg):
        self.latest_scan = msg

    def camera_callback(self, msg):
        """Chuyển đổi dữ liệu ảnh ROS byte sang OpenCV."""
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'bgr8':
                self.latest_image = img.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'rgb8':
                img_rgb = img.reshape((msg.height, msg.width, 3))
                self.latest_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            rospy.logerr(f"Lỗi nhận ảnh camera: {e}")

    def _set_state(self, new_state):
        rospy.loginfo(f"[STATE CHANGE] {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.state_change_time = rospy.get_time()

    # =========================================================================
    # XỬ LÝ ẢNH BÁM LÀN TRỰC QUAN
    # =========================================================================
    def get_lane_centers(self, frame):
        """Tìm tâm lòng đường dựa trên việc phát hiện 2 vạch biên trắng."""
        resized = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
        
        y_near = self.ROI_Y
        y_far = self.LOOKAHEAD_ROI_Y

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Lọc ngưỡng xám động (Adaptive Thresholding bằng phân vị để thích ứng ánh sáng thay đổi)
        dynamic_thresh_val = max(110, min(220, int(np.percentile(gray, 92))))
        _, thresh = cv2.threshold(gray, dynamic_thresh_val, 255, cv2.THRESH_BINARY)

        def find_borders(y_line):
            mid_x = int(self.WIDTH / 2)
            left_border = 0
            right_border = self.WIDTH - 1

            for x in range(mid_x, 0, -1):
                if thresh[y_line, x] == 255:
                    left_border = x
                    break
            for x in range(mid_x, self.WIDTH):
                if thresh[y_line, x] == 255:
                    right_border = x
                    break
            
            center_x = int((left_border + right_border) / 2)
            # Kiểm tra xem có thực sự thấy vạch kẻ ở dòng quét này không
            has_line = (left_border > 5 or right_border < self.WIDTH - 6)
            return center_x, left_border, right_border, has_line

        C_near, L_near, R_near, has_near = find_borders(y_near)
        C_far, L_far, R_far, has_far = find_borders(y_far)

        debug_frame = resized.copy()
        cv2.line(debug_frame, (0, y_near), (self.WIDTH, y_near), (0, 255, 255), 1)
        cv2.line(debug_frame, (0, y_far), (self.WIDTH, y_far), (0, 255, 255), 1)
        cv2.circle(debug_frame, (L_near, y_near), 5, (0, 0, 255), -1)
        cv2.circle(debug_frame, (R_near, y_near), 5, (0, 0, 255), -1)
        cv2.circle(debug_frame, (C_near, y_near), 6, (0, 255, 0), -1)

        return C_near, C_far, has_near, has_far, debug_frame

    # =========================================================================
    # NHẬN DIỆN BIỂN BÁO QUA ROBOFLOW API
    # =========================================================================
    def detect_traffic_sign(self):
        """
        Gửi ảnh camera hiện tại lên Roboflow API để phát hiện biển báo.
        Trả về nhãn biển báo ('L' - Rẽ Trái, 'R' - Rẽ Phải, 'F' - Đi Thẳng, 'Stop' - Dừng xe).
        """
        if self.latest_image is None:
            return None

        if not self.RF_API_KEY:
            rospy.logwarn("Chưa cấu hình ROBOFLOW_API_KEY! Trả về đi thẳng mặc định.")
            return 'F'

        try:
            # Lưu tạm ảnh thành file jpg để gửi API
            _, img_encoded = cv2.imencode('.jpg', self.latest_image)
            img_bytes = img_encoded.tobytes()

            upload_url = f"https://detect.roboflow.com/{self.RF_MODEL}/{self.RF_VERSION}"
            params = {"api_key": self.RF_API_KEY}
            
            rospy.loginfo("Đang gửi ảnh lên Roboflow API để quét biển báo...")
            response = requests.post(upload_url, params=params, data=img_bytes, timeout=5)
            result = response.json()

            predictions = result.get("predictions", [])
            if not predictions:
                rospy.loginfo("Roboflow không phát hiện biển báo nào.")
                return None

            # Lấy biển báo có độ tin cậy cao nhất
            best_pred = max(predictions, key=lambda x: x.get("confidence", 0.0))
            class_name = best_pred.get("class")
            conf = best_pred.get("confidence", 0.0)
            
            rospy.loginfo(f"Phát hiện biển báo: '{class_name}' với độ tin cậy: {conf:.2f}")

            # Gửi dữ liệu nhận diện lên Server BTC
            self.submit_detection_to_server(class_name, conf)
            
            # Map nhãn biển báo
            if class_name in ['L', 'turn_left', 'trai']:
                return 'L'
            elif class_name in ['R', 'turn_right', 'phai']:
                return 'R'
            elif class_name in ['F', 'straight', 'thang']:
                return 'F'
            elif class_name in ['Stop', 'stop', 'dung']:
                return 'Stop'
            
            return class_name

        except Exception as e:
            rospy.logerr(f"Lỗi khi gọi Roboflow API: {e}")
            return None

    def submit_detection_to_server(self, class_name, confidence):
        """Gửi gói tin sự kiện phát hiện biển báo về REST API của Server cuộc thi."""
        payload = {
            "race": 2,  # Smart City
            "team": self.TEAM_NAME,
            "detected_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace('+00:00','Z'),
            "sign_type": class_name,
            "confidence": float(confidence)
        }
        try:
            headers = {'Content-Type': 'application/json'}
            r = requests.post(self.SUBMIT_URL, json=payload, headers=headers, timeout=3)
            rospy.loginfo(f"Đã gửi báo cáo lên Server BTC. Kết quả Server: {r.status_code}")
        except Exception as e:
            rospy.logwarn(f"Không thể gửi báo cáo lên server BTC: {e}")

    def publish_mqtt_event(self, action):
        """Gửi MQTT báo cáo hướng rẽ của xe lên Broker."""
        if self.mqtt_client is None:
            return
        payload = {
            "team": self.TEAM_NAME,
            "timestamp": int(time.time()),
            "action": action  # 'left', 'right', 'straight', 'stop'
        }
        try:
            import json
            self.mqtt_client.publish(self.MQTT_DATA_TOPIC, json.dumps(payload))
            rospy.loginfo(f"MQTT Publish: {payload}")
        except Exception as e:
            rospy.logwarn(f"Lỗi MQTT Publish: {e}")

    # =========================================================================
    # VÒNG LẶP CHẠY XE THÀNH PHỐ THÔNG MINH
    # =========================================================================
    def run(self):
        rate = rospy.Rate(20)
        decision = None

        while not rospy.is_shutdown():
            if self.latest_image is None:
                rospy.logwarn_throttle(5, "Chờ hình ảnh từ camera...")
                rate.sleep()
                continue

            # 1. Quét tìm làn đường biên trắng
            C_near, C_far, has_near, has_far, debug_img = self.get_lane_centers(self.latest_image)

            # --- TRẠNG THÁI 1: BÁM LÀN THÀNH PHỐ THÔNG MINH ---
            if self.state == RobotState.STATE_LINE_FOLLOWING:
                # Điều khiển xe đi thẳng bằng bộ điều khiển P bám làn biên
                error_px = C_near - (self.WIDTH / 2.0)
                Kp = 0.007
                steering = max(-1.0, min(1.0, error_px * Kp))
                self.racer.steer(steering, self.BASE_SPEED)

                # Điều kiện chuyển sang giao lộ: vạch ở dòng quét xa (55%) bị đứt hoặc mất hẳn
                if not has_far:
                    rospy.loginfo("⚠️ Phát hiện vạch kẻ phía xa biến mất. Tiến hành tiếp cận giao lộ!")
                    self._set_state(RobotState.STATE_APPROACHING)

            # --- TRẠNG THÁI 2: ĐANG TIẾN VÀO GIAO LỘ ---
            elif self.state == RobotState.STATE_APPROACHING:
                # Bò ga chậm tiến sâu vào giữa ngã tư
                self.racer.steer(0.0, self.BASE_SPEED)
                
                # Sau thời gian cài đặt tiến ngã tư, xe dừng lại
                if rospy.get_time() - self.state_change_time > self.INTERSECTION_APPROACH_DURATION:
                    rospy.loginfo("Đã vào giữa giao lộ. Dừng xe quét biển báo.")
                    self.racer.stop()
                    time.sleep(0.5)
                    self._set_state(RobotState.STATE_HANDLING_SIGN)

            # --- TRẠNG THÁI 3: QUÉT BIỂN BÁO & QUYẾT ĐỊNH ---
            elif self.state == RobotState.STATE_HANDLING_SIGN:
                # Thực hiện quét gọi Roboflow API
                decision = self.detect_traffic_sign()
                
                if decision is None:
                    rospy.logwarn("Không đọc được biển báo! Thử lại sau 0.5s...")
                    time.sleep(0.5)
                    continue

                if decision == 'Stop':
                    rospy.loginfo("Gặp biển báo STOP. Hoàn thành nhiệm vụ!")
                    self.publish_mqtt_event('stop')
                    self._set_state(RobotState.STATE_GOAL_REACHED)
                    continue

                rospy.loginfo(f"Quyết định hướng đi tại giao lộ: {decision}")
                self._set_state(RobotState.STATE_TURNING)

            # --- TRẠNG THÁI 4: THỰC THI ĐÁNH LÁI QUA NGÃ TƯ ---
            elif self.state == RobotState.STATE_TURNING:
                if decision == 'L':
                    rospy.loginfo("Rẽ TRÁI 90 độ...")
                    self.publish_mqtt_event('left')
                    self.racer.turn_angle(-90)  # Rẽ trái góc 90 độ
                elif decision == 'R':
                    rospy.loginfo("Rẽ PHẢI 90 độ...")
                    self.publish_mqtt_event('right')
                    self.racer.turn_angle(90)   # Rẽ phải góc 90 độ
                else:
                    rospy.loginfo("Đi THẲNG qua giao lộ...")
                    self.publish_mqtt_event('straight')
                    # Đi thẳng bò qua giao lộ trong 1.2s
                    self.racer.steer(0.0, self.BASE_SPEED)
                    time.sleep(1.2)
                    self.racer.stop()

                # Đánh lái xong, chuyển sang tìm lại vạch làn đường mới
                self._set_state(RobotState.STATE_REACQUIRING)

            # --- TRẠNG THÁI 5: TÌM LẠI LÀN ĐƯỜNG MỚI ---
            elif self.state == RobotState.STATE_REACQUIRING:
                # Bò ga thẳng về trước để camera tìm thấy vạch
                self.racer.steer(0.0, self.BASE_SPEED)
                
                # Nếu camera tìm thấy vạch biên trắng an toàn
                if has_near:
                    rospy.loginfo("Đã bắt lại được vạch đường mới! Tiếp tục bám làn.")
                    self._set_state(RobotState.STATE_LINE_FOLLOWING)
                    continue

                # Quá thời gian timeout mà không bắt được vạch -> Dừng xe tránh đâm tường
                if rospy.get_time() - self.state_change_time > self.LINE_REACQUIRE_TIMEOUT:
                    rospy.logerr("Quá thời gian tìm làn đường mới. Dừng khẩn cấp!")
                    self._set_state(RobotState.STATE_DEAD_END)

            # --- TRẠNG THÁI 6: ĐÃ ĐẾN ĐÍCH ---
            elif self.state == RobotState.STATE_GOAL_REACHED:
                self.racer.stop()
                rospy.loginfo("=== ĐÃ VỀ ĐÍCH AN TOÀN ===")
                break

            # --- TRẠNG THÁI 7: LỖI HỆ THỐNG ---
            elif self.state == RobotState.STATE_DEAD_END:
                self.racer.stop()
                rospy.logerr("Dừng xe do lỗi hệ thống.")
                break

            # Ghi video debug
            if self.video_writer is not None:
                cv2.putText(debug_img, f"State: {self.state.name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                if decision:
                    cv2.putText(debug_img, f"Sign: {decision}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                self.video_writer.write(debug_img)

            rate.sleep()

        self.racer.stop()
        if self.video_writer is not None:
            self.video_writer.release()

if __name__ == '__main__':
    try:
        controller = SmartCityController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"Lỗi khẩn cấp Smart City: {e}")
        try:
            r = RacerController()
            r.stop()
        except:
            pass