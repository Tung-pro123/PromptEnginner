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

from src.core.control.racer_controller import RacerController
import onnxruntime as ort
from pyzbar.pyzbar import decode
import paho.mqtt.client as mqtt
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
        rospy.loginfo("Đăng ký nhận dữ liệu từ LiDAR và Camera thành công.")

    def setup_parameters(self):
        """Cấu hình các tham số hoạt động cho sa bàn thành phố thông minh"""
        self.BASE_SPEED = 0.16         # Tốc độ ga cơ bản của Smart City (chạy cẩn thận)
        self.WIDTH = 300
        self.HEIGHT = 300

        # Tham số thời gian chuyển trạng thái giao lộ
        self.INTERSECTION_APPROACH_DURATION = 0.6  # Thời gian bò thêm vào giữa giao lộ (giây)
        self.LINE_REACQUIRE_TIMEOUT = 3.0          # Quá thời gian này không tìm thấy vạch -> Dừng xe

        # Tọa độ quét vạch Camera
        self.ROI_Y = int(self.HEIGHT * 0.85)
        self.ROI_H = int(self.HEIGHT * 0.15)
        self.LOOKAHEAD_ROI_Y = int(self.HEIGHT * 0.55)
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
            self.robot = RacerController()
            rospy.loginfo("Phần cứng JetRacer (RacerController) đã được khởi tạo.")
        except Exception as e:
            rospy.logwarn(f"Không thể khởi tạo RacerController, sử dụng Mock object. Lỗi: {e}")
            from unittest.mock import Mock
            self.robot = Mock()

    def initialize_yolo(self):
        """Setup for Roboflow API (used instead of local YOLO ONNX model)."""
        # We keep YOLO session variable for backward compatibility but we won't load ONNX here.
        self.yolo_session = None
        # Attempt to locate submit script
        self.submit_module = self._load_submit_module()

    def _load_submit_module(self):
        """Try to import local 'submit_sign copy.py' and expose a submit function if available."""
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, self.SUBMIT_SCRIPT_FILENAME)
        if not os.path.exists(path):
            rospy.loginfo(f"Submit script not found at {path}, will fallback to HTTP submit to {self.SUBMIT_URL}")
            return None
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
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

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

                # --- BƯỚC 1: KIỂM TRA TÍN HIỆU ƯU TIÊN CAO (LiDAR) ---
                # Đây là tín hiệu đáng tin cậy nhất, nếu nó kích hoạt, xử lý ngay.
                if self.detector.process_detection():
                    rospy.loginfo("SỰ KIỆN (LiDAR): Phát hiện giao lộ. Dừng ngay lập tức.")
                    self.robot.stop()
                    time.sleep(0.5) # Chờ robot dừng hẳn

                    # Cập nhật vị trí hiện tại (đã đến đích) và xử lý
                    self.current_node_id = self.target_node_id
                    rospy.loginfo(f"==> ĐÃ ĐẾN node {self.current_node_id}.")

                    if self.current_node_id == self.navigator.end_node:
                        rospy.loginfo("ĐÃ ĐẾN ĐÍCH CUỐI CÙNG!")
                        self._set_state(RobotState.GOAL_REACHED)
                    else:
                        self._set_state(RobotState.HANDLING_EVENT)
                        self.handle_intersection()
                    continue # Bắt đầu vòng lặp mới với trạng thái mới

                # --- BƯỚC 2: LOGIC "NHÌN XA HƠN" VỚI ROI DỰ BÁO ---
                lookahead_line_center = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)

                if lookahead_line_center is None:
                    rospy.logwarn("SỰ KIỆN (Dự báo): Vạch kẻ đường biến mất ở phía xa. Chuẩn bị vào giao lộ.")
                    # Hành động phòng ngừa: chuyển sang trạng thái đi thẳng vào giao lộ.
                    self._set_state(RobotState.APPROACHING_INTERSECTION)
                    continue # Bắt đầu vòng lặp mới với trạng thái mới

                # --- BƯỚC 3: BÁM LINE BÌNH THƯỜNG (NẾU PHÍA TRƯỚC AN TOÀN) ---
                execution_line_center = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)

                if execution_line_center is not None:
                    # An toàn để bám line, vì chúng ta biết phía trước không có giao lộ đột ngột.
                    self.correct_course(execution_line_center)
                else:
                    # Trường hợp hiếm: ROI xa thấy line nhưng ROI gần lại không. Dừng lại cho an toàn.
                    rospy.logwarn("Trạng thái không nhất quán: ROI xa thấy line, ROI gần không thấy. Tạm dừng an toàn.")
                    self.robot.stop()

            # ===================================================================
            # TRẠNG THÁI 2: ĐANG TIẾN VÀO GIAO LỘ (APPROACHING_INTERSECTION)
            # ===================================================================
            elif self.current_state == RobotState.APPROACHING_INTERSECTION:
                # Đi thẳng một đoạn ngắn để vào trung tâm giao lộ
                self.robot.forward(self.BASE_SPEED)
                
                if rospy.get_time() - self.state_change_time > self.INTERSECTION_APPROACH_DURATION:
                    rospy.loginfo("Đã tiến vào trung tâm giao lộ. Dừng lại để xử lý.")
                    self.robot.stop() 
                    time.sleep(0.5)

                    self.current_node_id = self.target_node_id
                    rospy.loginfo(f"==> ĐÃ ĐẾN node {self.current_node_id}.")

                    if self.current_node_id == self.navigator.end_node:
                        rospy.loginfo("ĐÃ ĐẾN ĐÍCH CUỐI CÙNG!")
                        self._set_state(RobotState.GOAL_REACHED)
                    else:
                        self._set_state(RobotState.HANDLING_EVENT)
                        self.handle_intersection()

            # ===================================================================
            # TRẠNG THÁI 3: ĐANG RỜI KHỎI GIAO LỘ (LEAVING_INTERSECTION)
            # ===================================================================
            elif self.current_state == RobotState.LEAVING_INTERSECTION:
                self.robot.forward(self.BASE_SPEED)
                if rospy.get_time() - self.state_change_time > self.INTERSECTION_CLEARANCE_DURATION:
                    rospy.loginfo("Đã thoát khỏi khu vực giao lộ. Bắt đầu tìm kiếm line mới.")
                    self._set_state(RobotState.REACQUIRING_LINE)
            
            # ===================================================================
            # TRẠNG THÁI 4: ĐANG TÌM LẠI LINE (REACQUIRING_LINE)
            # ===================================================================
            elif self.current_state == RobotState.REACQUIRING_LINE:
                self.robot.forward(self.BASE_SPEED)
                line_center_x = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)
                
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

    def map_absolute_to_relative(self, target_direction_label, current_robot_direction):
        """
        Chuyển đổi hướng tuyệt đối ('N', 'E', 'S', 'W') thành hành động tương đối ('straight', 'left', 'right').
        Ví dụ: robot đang hướng BẮC (NORTH), mục tiêu là đi hướng ĐÔNG (EAST) -> hành động là 'right'.
        """
        target_dir = self.LABEL_TO_DIRECTION_ENUM.get(target_direction_label)
        if target_dir is None: return None

        current_idx = current_robot_direction.value
        target_idx = target_dir.value
        
        diff = (target_idx - current_idx + 4) % 4 
        
        if diff == 0:
            return 'straight'
        elif diff == 1:
            return 'right'
        elif diff == 3: 
            return 'left'
        else: 
            return 'turn_around'
        
    def map_relative_to_absolute(self, relative_action, current_robot_direction):
        """
        Chuyển đổi hành động tương đối ('straight', 'left', 'right') thành hướng tuyệt đối ('N', 'E', 'S', 'W').
        """
        current_idx = current_robot_direction.value
        if relative_action == 'straight':
            target_idx = current_idx
        elif relative_action == 'right':
            target_idx = (current_idx + 1) % 4
        elif relative_action == 'left':
            target_idx = (current_idx - 1 + 4) % 4
        else:
            return None
        
        for label, direction in self.LABEL_TO_DIRECTION_ENUM.items():
            if direction.value == target_idx:
                return label
        return None
    
    def _get_line_center(self, image, roi_y, roi_h):
        """Kiểm tra sự tồn tại và vị trí của vạch kẻ trong một ROI cụ thể."""
        if image is None: return None
        roi = image[roi_y : roi_y + roi_h, :]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Bước 1: Tạo mặt nạ màu sắc như cũ
        color_mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        
        # === BƯỚC 2: TẠO MẶT NẠ TẬP TRUNG (FOCUS MASK) ===
        focus_mask = np.zeros_like(color_mask)
        roi_height, roi_width = focus_mask.shape
        
        center_width = int(roi_width * self.ROI_CENTER_WIDTH_PERCENT)
        start_x = (roi_width - center_width) // 2
        end_x = start_x + center_width
        
        # Vẽ một hình chữ nhật trắng ở giữa
        cv2.rectangle(focus_mask, (start_x, 0), (end_x, roi_height), 255, -1)
        
        # === BƯỚC 3: KẾT HỢP HAI MẶT NẠ ===
        # Chỉ giữ lại những pixel trắng nào xuất hiện ở cả hai mặt nạ
        final_mask = cv2.bitwise_and(color_mask, focus_mask)

        # Tìm contours trên mặt nạ cuối cùng đã được lọc
        _, contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None
            
        c = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(c) < self.SCAN_PIXEL_THRESHOLD:
            return None

        M = cv2.moments(c)
        if M["m00"] > 0:
            # Quan trọng: Trọng tâm bây giờ được tính toán chỉ dựa trên vạch kẻ trong khu vực trung tâm
            return int(M["m10"] / M["m00"])
        return None
    
    def correct_course(self, line_center_x):
        """
        Hàm bám line an toàn với cơ chế giới hạn lực bẻ lái.
        """
        error = line_center_x - (self.WIDTH / 2)
        
        # Vẫn đi thẳng nếu sai số rất nhỏ
        if abs(error) < (self.WIDTH / 2) * self.SAFE_ZONE_PERCENT:
            self.robot.forward(self.BASE_SPEED)
            return

        # Tính toán lực điều chỉnh
        adj = (error / (self.WIDTH / 2)) * self.CORRECTION_GAIN

        # Ngăn chặn hành vi bẻ lái quá gắt một cách tuyệt đối
        adj = np.clip(adj, -self.MAX_CORRECTION_ADJ, self.MAX_CORRECTION_ADJ)
        
        # Áp dụng lực điều chỉnh đã được giới hạn (steer và chạy)
        self.robot.steer(adj, self.BASE_SPEED)
        
    def _submit_payload(self, payload):
        """
        Use the local submit script if available; otherwise perform HTTP POST to SUBMIT_URL.
        Expect server to return JSON with status code 201 on success.
        """
        try:
            if self.submit_module is not None:
                # Try common function names in submit script
                for fn in ('submit', 'submit_detection', 'main'):
                    if hasattr(self.submit_module, fn):
                        func = getattr(self.submit_module, fn)
                        try:
                            resp = func(payload)
                            rospy.loginfo(f"Submit script returned: {resp}")
                            return resp
                        except Exception as e:
                            rospy.logwarn(f"Call to submit script function '{fn}' failed: {e}")
                rospy.logwarn("Submit module loaded but no callable submit function found; falling back to HTTP POST.")
            # fallback HTTP POST
            headers = {'Content-Type': 'application/json'}
            r = requests.post(self.SUBMIT_URL, json=payload, headers=headers, timeout=6)
            try:
                j = r.json()
            except Exception:
                j = {'text': r.text}
            result = {'status_code': r.status_code, 'response': j}
            rospy.loginfo(f"HTTP submit result: {result}")
            return result
        except Exception as e:
            rospy.logerr(f"Error submitting payload: {e}")
            return {'status_code': 0, 'response': str(e)}

    def handle_intersection(self):
        rospy.loginfo("\n[GIAO LỘ] Dừng lại và xử lý...")
        self.robot.stop() 
        time.sleep(0.5)

        current_direction = self.DIRECTIONS[self.current_direction_index]
        angle_to_sign = self.ANGLE_TO_FACE_SIGN_MAP.get(current_direction, 0)
        self.turn_robot(angle_to_sign, False)
        image_info = self.latest_image
        detections = self.detect_with_yolo(image_info)
        self.turn_robot(-angle_to_sign, False)
        
        prescriptive_cmds = {det['class_name'] for det in detections if det['class_name'] in self.PRESCRIPTIVE_SIGNS}
        prohibitive_cmds = {det['class_name'] for det in detections if det['class_name'] in self.PROHIBITIVE_SIGNS}
        data_items = [det for det in detections if det['class_name'] in self.DATA_ITEMS]

        # 2. Xử lý các mục dữ liệu (QR, Toán) và Publish
        rospy.loginfo("[STEP 2] Processing data items...")
        for item in data_items:
            if item['class_name'] == 'qr_code':
                rospy.loginfo("Found QR Code. Publishing data...")
                self.publish_data({'type': 'QR_CODE', 'value': 'simulated_data_123'})
            elif item['class_name'] == 'math_problem':
                rospy.loginfo("Found Math Problem. Solving and publishing...")
                self.publish_data({'type': 'MATH_PROBLEM', 'value': '2+2=4'})

        # Attempt to submit detected signs (prescriptive/prohibitive) to server via submit module or HTTP
        for det in detections:
            try:
                payload = {
                    "text": f"Detected: {det.get('class_name')}, conf={det.get('confidence')}",
                    "race": 1,
                    "node_id": int(self.current_node_id) if isinstance(self.current_node_id, (int,str)) and str(self.current_node_id).isdigit() else self.current_node_id,
                    "submit_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace('+00:00','Z'),
                    "team": self.TEAM_NAME,
                    "Map_type": self.MAP_FILE_PATH
                }
                rospy.loginfo(f"Submitting detection payload: {payload}")
                submit_result = self._submit_payload(payload)
                # If HTTP fallback returned structure, map to expected short log
                status = submit_result.get('status_code') if isinstance(submit_result, dict) else None
                if status == 201:
                    rospy.loginfo(f"Thành công (Status Code: 201) -> {submit_result.get('response')}")
                else:
                    rospy.loginfo(f"Submit returned: {submit_result}")
            except Exception as e:
                rospy.logerr(f"Error while submitting detection: {e}")

        rospy.loginfo("[STEP 3] Lập kế hoạch điều hướng theo bản đồ...")
        # 3. Lập kế hoạch Điều hướng
        final_decision = None
        is_deviation = False 

        while True:
            planned_direction_label = self.navigator.get_next_direction_label(self.current_node_id, self.planned_path)
            if not planned_direction_label:
                rospy.logerr("Lỗi kế hoạch: Không tìm thấy bước tiếp theo.") 
                self._set_state(RobotState.DEAD_END) 
                return
            
            planned_action = self.map_absolute_to_relative(planned_direction_label, current_direction)
            rospy.loginfo(f"Kế hoạch A* đề xuất: Đi {planned_action} (hướng {planned_direction_label})")

            # Ưu tiên 1: Biển báo bắt buộc
            intended_action = None
            if 'L' in prescriptive_cmds: intended_action = 'left'
            elif 'R' in prescriptive_cmds: intended_action = 'right'
            elif 'F' in prescriptive_cmds: intended_action = 'straight'
            
            # Ưu tiên 2: Plan
            if intended_action is None:
                intended_action = planned_action
            else:
                # Nếu hành động bắt buộc khác với kế hoạch, đánh dấu là đi chệch hướng
                if intended_action != planned_action:
                    is_deviation = True
                    rospy.logwarn(f"CHỆCH HƯỚNG! Biển báo bắt buộc ({intended_action}) khác với kế hoạch ({planned_action}).")

            # 3.3. Veto bởi biển báo cấm
            is_prohibited = (intended_action == 'straight' and 'NF' in prohibitive_cmds) or \
                            (intended_action == 'right' and 'NR' in prohibitive_cmds) or \
                            (intended_action == 'left' and 'NL' in prohibitive_cmds)

            if is_prohibited:
                rospy.logwarn(f"Hành động dự định '{intended_action}' bị CẤM!")
                
                # Nếu hành động bị cấm đến từ biển báo bắt buộc -> Lỗi bản đồ
                if is_deviation:
                    rospy.logerr("LỖI BẢN ĐỒ! Biển báo bắt buộc mâu thuẫn với biển báo cấm. Không thể đi tiếp.")
                    self._set_state(RobotState.DEAD_END) 
                    return
                
                # Nếu hành động bị cấm đến từ kế hoạch A* -> Tìm đường lại
                banned_edge = (self.current_node_id, self.planned_path[self.planned_path.index(self.current_node_id) + 1])
                if banned_edge not in self.banned_edges:
                    self.banned_edges.append(banned_edge)
                
                rospy.loginfo(f"Thêm cạnh cấm {banned_edge} và tìm đường lại...")
                # Use find_shortest_path_through_loads if available to respect loads requirement
                if hasattr(self.navigator, 'find_shortest_path_through_loads'):
                    new_path = self.navigator.find_shortest_path_through_loads(self.current_node_id, self.navigator.end_node, self.banned_edges)
                else:
                    new_path = self.navigator.find_path(self.current_node_id, self.navigator.end_node, self.banned_edges)

                if new_path:
                    self.planned_path = new_path
                    rospy.loginfo(f"Đã tìm thấy đường đi mới: {self.planned_path}")
                    continue # Quay lại đầu vòng lặp để kiểm tra với kế hoạch mới
                else:
                    rospy.logerr("Không thể tìm đường đi mới sau khi gặp biển cấm.")
                    self._set_state(RobotState.DEAD_END)
                    return
            
            final_decision = intended_action
            break 

        # 4. Thực thi quyết định
        if final_decision == 'straight': 
            rospy.loginfo("[FINAL] Decision: Go STRAIGHT.")
        elif final_decision == 'right': 
            rospy.loginfo("[FINAL] Decision: Turn RIGHT.") 
            self.turn_robot(90, True)
        elif final_decision == 'left': 
            rospy.loginfo("[FINAL] Decision: Turn LEFT.") 
            self.turn_robot(-90, True)
        else:
            rospy.logwarn("[!!!] DEAD END! No valid paths found.") 
            self._set_state(RobotState.DEAD_END)
            return
        
        # 5. Cập nhật trạng thái robot sau khi thực hiện
        # 5.1. Xác định node tiếp theo
        next_node_id = None
        if not is_deviation:
            # Nếu đi theo kế hoạch, chỉ cần lấy node tiếp theo từ path
            next_node_id = self.planned_path[self.planned_path.index(self.current_node_id) + 1]
        else:
            # Nếu chệch hướng, phải tìm node tiếp theo dựa trên hành động đã thực hiện
            new_robot_direction = self.DIRECTIONS[self.current_direction_index] 
            
            executed_direction_label = None
            for label, direction_enum in self.LABEL_TO_DIRECTION_ENUM.items():
                if direction_enum == new_robot_direction:
                    executed_direction_label = label 
                    break
            
            if executed_direction_label is None:
                rospy.logerr("Lỗi logic: Không thể tìm thấy label cho hướng đi mới của robot.") 
                self._set_state(RobotState.DEAD_END) 
                return

            next_node_id = self.navigator.get_neighbor_by_direction(self.current_node_id, executed_direction_label)
            if next_node_id is None:
                 rospy.logerr("LỖI BẢN ĐỒ! Đã thực hiện rẽ nhưng không có node tương ứng.")
                 self._set_state(RobotState.DEAD_END)
                 return
            
            # Quan trọng: Lập kế hoạch lại từ vị trí mới
            rospy.loginfo(f"Đã đi chệch kế hoạch. Lập lại đường đi từ node mới {next_node_id}...")
            if hasattr(self.navigator, 'find_shortest_path_through_loads'):
                new_path = self.navigator.find_shortest_path_through_loads(next_node_id, self.navigator.end_node, self.banned_edges)
            else:
                new_path = self.navigator.find_path(next_node_id, self.navigator.end_node, self.banned_edges)
            if new_path:
                self.planned_path = new_path
                rospy.loginfo(f"Đường đi mới sau khi chệch hướng: {self.planned_path}")
            else:
                rospy.logerr("Không thể tìm đường về đích từ vị trí mới.")
                self._set_state(RobotState.DEAD_END)
                return

        self.target_node_id = next_node_id
        rospy.loginfo(f"==> Đang di chuyển đến node tiếp theo: {self.target_node_id}")
        self._set_state(RobotState.LEAVING_INTERSECTION)
    
    def turn_robot(self, degrees, update_main_direction=True):
        # Trên JetRacer (Ackermann), ta rẽ theo cung tròn thông qua turn_angle
        # Hàm record_frame được truyền vào để tiếp tục ghi video trong khi rẽ
        self.robot.turn_angle(degrees, record_callback=self._record_frame)

        if update_main_direction and degrees % 90 == 0 and degrees != 0:
            num_turns = round(degrees / 90)
            self.current_direction_index = (self.current_direction_index + num_turns + 4) % 4
            rospy.loginfo(f"==> Hướng đi MỚI: {self.DIRECTIONS[self.current_direction_index].name}")
        time.sleep(0.5)
        self._record_frame()
    
    def _does_path_exist_in_frame(self, image):
        if image is None: return False
        roi = image[self.ROI_Y : self.ROI_Y + self.ROI_H, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        _img, contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return bool(contours) and cv2.contourArea(max(contours, key=cv2.contourArea)) > self.SCAN_PIXEL_THRESHOLD
    
    def scan_for_available_paths_proactive(self):
        rospy.loginfo("[SCAN] Bắt đầu quét chủ động...")
        paths = {"straight": False, "right": False, "left": False}
        if self.latest_image is not None:
            paths["straight"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(90, update_main_direction=False)
        time.sleep(0.5)
        if self.latest_image is not None:
            paths["right"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(-180, update_main_direction=False)
        time.sleep(0.5)
        if self.latest_image is not None:
            paths["left"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(90, update_main_direction=False)
        rospy.loginfo(f"[SCAN] Kết quả: {paths}")
        return paths

def main():
    rospy.init_node('jetbot_controller_node', anonymous=True)
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