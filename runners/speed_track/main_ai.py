#!/usr/bin/env python3
"""
Speed Track AI Controller - Blackboard Architecture
AI Mạng Neural (Lane Keeping) không dùng LiDAR
"""
import sys
# Hỗ trợ ROS Python path
py3 = [p for p in sys.path if 'python2.7' not in p]
py2 = [p for p in sys.path if 'python2.7' in p]
sys.path = py3 + py2

import os, time
# [FIX JETSON] Tự động thêm đường dẫn CUDA vào môi trường trước khi gọi ONNXRuntime
if "/usr/local/cuda/lib64" not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] Cần cài đặt onnxruntime. Chạy: pip install onnxruntime-gpu")
    sys.exit(1)
import numpy as np
import cv2
import rospy
from enum import Enum
from sensor_msgs.msg import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from robot.control.racer_controller import RacerController
from robot.debug.debugger import Debugger

# ============================================================
# ENUMS
# ============================================================
class TrackState(Enum):
    WAITING = 0; KEEP_LANE = 1
    CHECKPOINT_CD = 4; E_STOP = 5; FINISHED = 6

# ============================================================
# BLACKBOARD (BẢNG ĐEN)
# ============================================================
class Blackboard:
    def __init__(self):
        # 1. Dữ liệu Cảm Biến
        self.image = None
        
        # 2. Đầu ra của Mô đun AI (Vision)
        self.ai_steer = 0.0
        self.ai_throttle = 0.0
        self.ai_valid = False
        
        # 3. Đầu ra của Mô đun Checkpoint
        self.checkpoint_detected = False
        
        # 4. Dữ liệu Quản lý Trạng Thái (FSM/Arbiter)
        self.state = TrackState.WAITING
        self.state_time = 0.0
        self.cp_count = 0
        self.cp_last_time = 0.0
        
        # 5. Lệnh Điều Khiển Cuối Cùng
        self.cmd_steer = 0.0
        self.cmd_throttle = 0.0

    def get(self, key, default=None):
        """Hỗ trợ tương thích với class Debugger"""
        if key in ['latest_image', 'raw_camera_frame']:
            return self.image
        elif key == 'state_name':
            return self.state.name
        elif key == 'steering':
            return self.cmd_steer
        elif key == 'throttle':
            return self.cmd_throttle
        elif key == 'ai_action':
            return "ONNX_AI" if self.ai_valid else "WAIT"
        elif key == 'ai_steering':
            return self.ai_steer
        elif key == 'ai_throttle':
            return self.ai_throttle
        return getattr(self, key, default)

# ============================================================
# MAIN CONTROLLER (BLACKBOARD ARCHITECTURE)
# ============================================================
class SpeedTrackBlackboardController:
    def __init__(self):
        # --- Cấu hình Xe (Params) ---
        self.BASE_SPEED = 0.3

        # --- Khởi tạo Debugger ---
        self.debugger = Debugger(debug_mode=True)
        self.debugger._info("=== KHOI TAO SPEED TRACK AI (Khong Lidar) ===")
        
        self.bb = Blackboard()
        self.racer = RacerController()
        self.racer.stop()
        
        # --- Cấu hình AI ---
        self.IMG_WIDTH = 128
        self.IMG_HEIGHT = 32
        
        # --- Khởi tạo ONNXRuntime ---
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'vision_inference.onnx')
        model_path = os.path.abspath(model_path)
        
        try:
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.debugger._info(f"Đã load trọng số ONNX thành công! Providers: {self.session.get_providers()}")
        except Exception as e:
            self.debugger.log_error(e, f"Không thể load trọng số ONNX tại {model_path}")
            sys.exit(1)
            
        # --- Đăng ký ROS Subscribers ---
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self._cam_cb, queue_size=1)
        
        self.set_state(TrackState.WAITING)
        self.debugger._info("=== SAN SANG ===")

    # ============================================================
    # SENSOR CALLBACKS (Ghi dữ liệu lên Blackboard)
    # ============================================================
    def _cam_cb(self, msg):
        try:
            if 'compressed' in msg.encoding:
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
            self.bb.image = img
        except Exception as e:
            self.debugger.log_error(e, "Lỗi đọc camera")

    # ============================================================
    # MODULES WORKERS (Tính toán và cập nhật Blackboard)
    # ============================================================
    
    def update_ai_module(self):
        """Mô-đun Mắt: Xử lý ảnh và ném vào Neural Network"""
        if self.bb.image is None:
            self.bb.ai_valid = False
            return
            
        frame = self.bb.image
        h, w = frame.shape[:2]
        roi_y1 = int(144 * h / 480)
        roi_y2 = h
        roi_w = min(w, 640) 
        
        roi = frame[roi_y1:roi_y2, 0:roi_w]
        img = cv2.resize(roi, (self.IMG_WIDTH, self.IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        
        # Chuẩn bị dữ liệu cho ONNX (shape: 1, 1, H, W)
        img_array = np.expand_dims(np.expand_dims(img, axis=0), axis=0).astype(np.float32)
        
        try:
            outputs = self.session.run(None, {self.input_name: img_array})
            # outputs[0] thường có shape (1, 2)
            self.bb.ai_steer = float(outputs[0][0][0])
            # Giữ an toàn, dùng tốc độ mặc định thay vì phụ thuộc vào model
            self.bb.ai_throttle = self.BASE_SPEED
            self.bb.ai_valid = True
        except Exception as e:
            self.debugger.log_error(e, "Lỗi chạy inference ONNX")
            self.bb.ai_valid = False
            
        # Vẽ khung đỏ lên hình gốc (hiển thị vùng ROI cho Debugger)
        cv2.rectangle(self.bb.image, (0, roi_y1), (roi_w, roi_y2), (0, 0, 255), 2)
        cv2.putText(self.bb.image, "AI ROI", (10, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def update_checkpoint_module(self):
        """Mô-đun Checkpoint: Tìm vạch trắng ngang đường"""
        if self.bb.image is None: 
            self.bb.checkpoint_detected = False
            return
        
        H = self.bb.image.shape[0]
        CP_ROI_Y = int(H * 0.88)
        CP_ROI_H = int(H * 0.10)
        roi = self.bb.image[CP_ROI_Y:CP_ROI_Y+CP_ROI_H, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        ratio = (np.sum(b > 0) / b.size)
        self.bb.checkpoint_detected = (ratio >= 0.45)

    # ============================================================
    # ARBITER (Não bộ Phán Xử)
    # ============================================================
    def set_state(self, s):
        if self.bb.state != s:
            self.debugger._info(f"STATE: {self.bb.state.name} -> {s.name}")
            self.bb.state = s
            self.bb.state_time = rospy.get_time()

    def time_in_state(self):
        return rospy.get_time() - self.bb.state_time

    def run_arbiter(self):
        """Não bộ chỉ đọc dữ liệu để chuyển state phục vụ mục đích ghi log, điều khiển do AI quyết định hoàn toàn."""
        
        # 1. Trực tiếp lấy lệnh từ AI đưa xuống xe (Không để FSM can thiệp)
        if self.bb.ai_valid:
            self.bb.cmd_steer = max(-1.0, min(1.0, self.bb.ai_steer))
            self.bb.cmd_throttle = self.bb.ai_throttle
        else:
            self.bb.cmd_steer = 0.0
            self.bb.cmd_throttle = 0.0

        # 2. Cập nhật State Machine (Chỉ để phục vụ ghi Log / HUD)
        
        # --- WAITING ---
        if self.bb.state == TrackState.WAITING:
            if self.bb.ai_valid:
                self.set_state(TrackState.KEEP_LANE)
            elif self.time_in_state() > 30.0:
                self.set_state(TrackState.E_STOP)

        # --- KEEP LANE (AI Dẫn Đường) ---
        elif self.bb.state == TrackState.KEEP_LANE:
            # Kiểm tra chuyển trạng thái Checkpoint
            if self.bb.checkpoint_detected:
                now = time.time()
                if now - self.bb.cp_last_time > 3.0:
                    self.bb.cp_count += 1
                    self.bb.cp_last_time = now
                    self.debugger._info(f"*** CHECKPOINT {self.bb.cp_count} ***")
                    self.set_state(TrackState.CHECKPOINT_CD)

        # --- CHECKPOINT COOLDOWN ---
        elif self.bb.state == TrackState.CHECKPOINT_CD:
            if self.time_in_state() > 2.0:
                self.set_state(TrackState.KEEP_LANE)

        # --- E_STOP / FINISHED ---
        elif self.bb.state in [TrackState.E_STOP, TrackState.FINISHED]:
            # Chỉ khi chết hẳn (E_STOP/FINISHED) mới ngắt động cơ để đảm bảo an toàn
            self.bb.cmd_steer = 0.0
            self.bb.cmd_throttle = 0.0

    # ============================================================
    # MAIN LOOP
    # ============================================================
    def run(self):
        self.debugger._info("Đợi 3s để hệ thống khởi động...")
        time.sleep(3)
        self.debugger._info("=== BẮT ĐẦU ĐIỀU KHIỂN (Thuần AI) ===")
        
        rate = rospy.Rate(20) 
        
        while not rospy.is_shutdown():
            # 1. Các Module chạy độc lập tính toán và ghi kết quả lên Blackboard
            self.update_ai_module()
            self.update_checkpoint_module()
            
            # 2. NÃO BỘ đọc Blackboard để Ra Quyết Định Cuối Cùng
            self.run_arbiter()
            
            # 3. Ghi Debug (Video, CSV, Terminal)
            self.debugger.process(self.bb)
            
            # 4. Gửi lệnh xuống Actuator
            self.racer.steer(self.bb.cmd_steer, self.bb.cmd_throttle)
            
            # 5. Kiem tra E_STOP
            if self.bb.state in [TrackState.E_STOP, TrackState.FINISHED]:
                break
                
            rate.sleep()
            
        self.racer.stop()
        if hasattr(self, 'debugger'):
            self.debugger._info("Kết thúc.")
            self.debugger.close()

def main():
    rospy.init_node('speed_track_ai_controller', anonymous=True)
    try:
        controller = SpeedTrackBlackboardController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        try: RacerController().stop()
        except: pass

if __name__ == '__main__':
    main()
