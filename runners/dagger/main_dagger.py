#!/usr/bin/env python3
"""
DAgger Main Controller
======================
Hệ thống điều khiển xe tự hành kết hợp:
  - Bám làn (Line Following) từ Camera + LaneDetector V3
  - Né vật cản (Obstacle Avoidance) từ LiDAR 5-zone
  - Học tương tác trực tuyến (DAgger) qua Gamepad Joy

=== HƯỚNG DẪN SỬ DỤNG GAMEPAD (PS4/Xbox) ===

  [CHẾ ĐỘ VẬN HÀNH]
  - Mặc định: AI kiểm soát xe hoàn toàn (AI_CONTROL mode)
  - Nhích Left Stick (axes[0]) hoặc bóp R2/L2: CHUYỂN sang HUMAN mode
    (xe nhượng quyền cho Joy + tự động thu thập dữ liệu)
  - Thả Joy về giữa: trả quyền lại cho AI sau JOY_RELEASE_TIMEOUT giây

  [PHÍM CHỨC NĂNG]
  - Triangle (button 3) : Bắt đầu xe / Mở khóa E-STOP
  - Circle   (button 1) : E-STOP khẩn cấp (khoá động cơ)
  - L1       (button 4) : Lưu model ngay lập tức
  - R1       (button 5) : Turbo mode (tăng speed cap)
  - D-pad ↑  (axes[7]=1): Tăng tốc độ cơ bản +0.05
  - D-pad ↓  (axes[7]=-1): Giảm tốc độ cơ bản -0.05

  [THU THẬP DỮ LIỆU]
  - Bất kỳ khi nào Joy đang active → hệ thống TỰ ĐỘNG ghi (S_t, A_joy)
  - Không cần bấm nút đặc biệt — chỉ cần cầm Joy và can thiệp là đủ
  - Số mẫu đã thu + loss được hiển thị trên console

  [ĐƯỜNG DẪN]
  - Model: models/dagger_policy.pt
  - Anchor CSV: logs/dagger/anchor_data.csv (nếu có)
  - Log CSV mỗi session: logs/dagger/session_YYYYMMDD_HHMMSS.csv

=== KIẾN TRÚC ===

  Main Loop (30 Hz)
    ├── [Sensor] Camera → LaneDetector → LaneStateEstimator
    ├── [Sensor] LiDAR → extract_lidar_zones()
    ├── [State]  build S_t = extract_state(lane_state, scan)
    ├── [Arbiter] Joy active? → HUMAN mode : AI mode
    │     ├── HUMAN: A_t = Joy input, push(S_t, A_t) → ReplayBuffer
    │     └── AI   : A_t = Policy.predict(S_t)
    ├── [Safety] SafetyLayer.check(A_t, lidar_zones)
    └── [Actuator] RacerController.steer(steer, throttle)

  Background Thread
    └── BackgroundTrainer: sample ReplayBuffer → Policy.update()
"""

import sys, os
py3 = [p for p in sys.path if 'python2.7' not in p]
py2 = [p for p in sys.path if 'python2.7' in p]
sys.path = py3 + py2

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _BASE)

import time
import math
import csv
import threading

import rospy
import numpy as np
import cv2
from enum import Enum
from sensor_msgs.msg import Joy, LaserScan, Image

# --- Hardware ---
from robot.control.racer_controller import RacerController

# --- Perception (V3 pipeline) ---
from robot.config.v3_config          import V3Config
from robot.perception.bev  import BEVTransform
from robot.perception.lane_detector_v3 import MultiLaneDetector
from robot.estimation.geometry  import LaneGeometry
from robot.estimation.lane_state import LaneStateEstimator

# --- DAgger components ---
from robot.dagger.state_extractor import extract_state, STATE_DIM
from robot.dagger.replay_buffer   import ReplayBuffer
from robot.dagger.policy          import DAggerPolicy
from robot.dagger.trainer         import BackgroundTrainer
from robot.dagger.safety          import SafetyLayer

# =====================================================================
# PATHS
# =====================================================================
_ROOT       = _BASE
MODEL_PATH  = os.path.join(_ROOT, 'models', 'dagger_policy.pt')
LOG_DIR     = os.path.join(_ROOT, 'logs', 'dagger')
ANCHOR_CSV  = os.path.join(LOG_DIR, 'anchor_data.csv')

# =====================================================================
# PARAMS
# =====================================================================
LOOP_RATE_HZ        = 30
BASE_SPEED          = 0.25    # throttle cơ bản khi AI lái thẳng
MAX_SPEED           = 0.45    # khi Turbo
JOY_DEAD_ZONE       = 0.08    # ngưỡng coi Joy đang được chạm
JOY_RELEASE_TIMEOUT = 0.5     # giây sau khi thả Joy mới trả quyền về AI
AUTOSAVE_EVERY      = 300     # tự lưu model mỗi N Joy-sample mới
PRINT_STATUS_EVERY  = 30      # in status mỗi N vòng loop (= ~1s ở 30Hz)

# =====================================================================
# STATE MACHINE
# =====================================================================
class DAggerState(Enum):
    WAITING          = 0   # chờ khởi động (xe đứng yên)
    AI_CONTROL       = 1   # AI lái
    HUMAN_CONTROL    = 2   # Người lái (Joy can thiệp) + thu thập data
    SAFETY_STOP      = 3   # Safety layer phanh khẩn cấp
    E_STOP           = 4   # Khóa motor hoàn toàn

# =====================================================================
# MAIN CONTROLLER
# =====================================================================
class DAggerController:
    def __init__(self):
        rospy.loginfo("=== KHOI TAO DAGGER CONTROLLER ===")

        # --- Config & perception pipeline ---
        self.cfg         = V3Config()
        self.bev         = BEVTransform(self.cfg)
        self.detector    = MultiLaneDetector(self.cfg)
        self.geometry    = LaneGeometry(self.cfg, self.bev)
        self.estimator   = LaneStateEstimator(self.cfg, self.bev)

        # --- Hardware ---
        self.racer = RacerController()
        self.racer.stop()

        # --- DAgger components ---
        os.makedirs(LOG_DIR, exist_ok=True)
        self.buffer  = ReplayBuffer(anchor_csv_path=ANCHOR_CSV)
        self.policy  = DAggerPolicy(model_path=MODEL_PATH)
        self.trainer = BackgroundTrainer(self.policy, self.buffer)
        self.safety  = SafetyLayer()

        # --- State machine ---
        self.state      = DAggerState.WAITING
        self.state_time = rospy.get_time()

        # --- Sensor data (thread-safe với GIL đủ cho numpy assign) ---
        self.latest_image = None
        self.latest_scan  = None
        self.lane_state   = None

        # --- Joy state ---
        self.joy_steer        = 0.0
        self.joy_throttle     = 0.0
        self.joy_active       = False      # Joy đang được chạm
        self.joy_last_active  = 0.0        # timestamp lần cuối Joy active
        self.turbo_mode       = False
        self.e_stop_locked    = True       # bắt đầu ở trạng thái khoá (an toàn)
        self.speed_trim       = 0.0
        self.dpad_up_pressed  = False
        self.dpad_down_pressed = False

        # --- Logging ---
        ts       = time.strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(LOG_DIR, f'session_{ts}.csv')
        self._log_file = open(log_path, 'w', newline='')
        self._csv = csv.writer(self._log_file)
        self._csv.writerow([
            'timestamp', 'mode', 'state',
            'e_y', 'theta_e', 'line_visible',
            'd_left', 'd_front_left', 'd_front', 'd_front_right', 'd_right',
            'obstacle_detected',
            'cmd_steer', 'cmd_throttle', 'is_safety_stop',
            'buffer_size', 'train_updates'
        ])

        self._loop_count    = 0
        self._joy_samples   = 0
        self._last_autosave = 0

        # --- ROS subscribers ---
        rospy.Subscriber(self.cfg.camera_topic, Image, self._cam_cb, queue_size=1)
        rospy.Subscriber(self.cfg.lidar_topic,  LaserScan, self._lidar_cb, queue_size=1)
        rospy.Subscriber('/joy',               Joy,       self._joy_cb,   queue_size=1)

        rospy.loginfo("=== SAN SANG — Nhan Triangle (button 3) de bat dau ===")
        self._print_controls()

    # ==================================================================
    # ROS CALLBACKS
    # ==================================================================

    def _cam_cb(self, msg):
        try:
            if 'compressed' in msg.encoding:
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if 'rgb' in msg.encoding else img
            self.latest_image = img
        except Exception as e:
            rospy.logerr_throttle(5, f"[DAgger] Cam error: {e}")

    def _lidar_cb(self, msg):
        self.latest_scan = msg

    def _joy_cb(self, msg):
        """Xử lý tất cả tín hiệu tay cầm."""
        try:
            axes    = msg.axes
            buttons = msg.buttons

            # ---- E-STOP (Circle = button 1) ----
            if len(buttons) > 1 and buttons[1] == 1:
                self.e_stop_locked = True
                self._set_state(DAggerState.E_STOP)
                rospy.logwarn("[DAgger] >>> E-STOP KICH HOAT <<<")

            # ---- Mở khoá (Triangle = button 3) ----
            if len(buttons) > 3 and buttons[3] == 1:
                self.e_stop_locked = False
                if self.state == DAggerState.E_STOP:
                    self._set_state(DAggerState.WAITING)
                rospy.loginfo("[DAgger] >>> MO KHOA — xe san sang chay <<<")

            # ---- Lưu model ngay (L1 = button 4) ----
            if len(buttons) > 4 and buttons[4] == 1:
                self.policy.save(MODEL_PATH)
                self.buffer.save_csv(ANCHOR_CSV)
                rospy.loginfo("[DAgger] Đã lưu model + buffer theo lệnh tay.")

            # ---- Turbo mode (R1 = button 5) ----
            self.turbo_mode = (len(buttons) > 5 and buttons[5] == 1)

            # ---- Speed trim (D-pad ↑↓ = axes[7]) ----
            if len(axes) > 7:
                dpad_y = axes[7]
                if dpad_y == 1.0 and not self.dpad_up_pressed:
                    self.speed_trim = min(self.speed_trim + 0.05, 0.20)
                    self.dpad_up_pressed = True
                    rospy.loginfo(f"[DAgger] Speed trim: {BASE_SPEED + self.speed_trim:.2f}")
                elif dpad_y == -1.0 and not self.dpad_down_pressed:
                    self.speed_trim = max(self.speed_trim - 0.05, -0.10)
                    self.dpad_down_pressed = True
                    rospy.loginfo(f"[DAgger] Speed trim: {BASE_SPEED + self.speed_trim:.2f}")
                elif dpad_y == 0.0:
                    self.dpad_up_pressed  = False
                    self.dpad_down_pressed = False

            # ---- Throttle (R2/L2 = axes[5]/axes[2]) ----
            r2 = (1.0 - axes[5]) / 2.0 if len(axes) > 5 else 0.0
            l2 = (1.0 - axes[2]) / 2.0 if len(axes) > 2 else 0.0
            raw_throttle = r2 - l2      # forward: +, backward: -

            # ---- Steer (Left Stick X = axes[0]) ----
            raw_steer = axes[0] if len(axes) > 0 else 0.0

            # ---- Xác định Joy có đang active không ----
            steer_active    = abs(raw_steer)    > JOY_DEAD_ZONE
            throttle_active = abs(raw_throttle) > JOY_DEAD_ZONE

            if steer_active or throttle_active:
                self.joy_active      = True
                self.joy_last_active = rospy.get_time()
                # Clamp và gán
                spd_cap = MAX_SPEED if self.turbo_mode else (BASE_SPEED + self.speed_trim)
                self.joy_throttle = max(-spd_cap, min(spd_cap, raw_throttle * spd_cap))
                self.joy_steer    = max(-1.0, min(1.0, raw_steer))
            else:
                # Thả Joy → chờ timeout rồi mới trả AI
                if rospy.get_time() - self.joy_last_active > JOY_RELEASE_TIMEOUT:
                    self.joy_active   = False
                    self.joy_steer    = 0.0
                    self.joy_throttle = 0.0

        except Exception as e:
            rospy.logerr_throttle(3, f"[DAgger] Joy error: {e}")

    # ==================================================================
    # PERCEPTION UPDATE
    # ==================================================================

    def _update_perception(self):
        """Chạy lane detection pipeline V3 để cập nhật self.lane_state."""
        if self.latest_image is None:
            return

        try:
            img = self.latest_image.copy()

            # BEV transform
            bev_img = self.bev.transform(img)

            # Segmentation (HSV mask)
            hsv  = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
            # Dùng cùng màu HSV với main_speed_track_v3
            m1   = cv2.inRange(hsv, np.array([0, 80, 80]),   np.array([18, 255, 255]))
            m2   = cv2.inRange(hsv, np.array([155, 80, 80]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(m1, m2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

            # Lane detection → geometry → state estimation
            det_result = self.detector.detect(mask)
            obs        = self.geometry.process(det_result, self.lane_state)
            self.lane_state = self.estimator.update(obs)

        except Exception as e:
            rospy.logerr_throttle(5, f"[DAgger] Perception error: {e}")

    # ==================================================================
    # STATE MACHINE
    # ==================================================================

    def _set_state(self, new_state):
        if self.state != new_state:
            rospy.loginfo(f"[DAgger] STATE: {self.state.name} → {new_state.name}")
            self.state      = new_state
            self.state_time = rospy.get_time()

    def _time_in_state(self):
        return rospy.get_time() - self.state_time

    # ==================================================================
    # MAIN LOOP
    # ==================================================================

    def run(self):
        rospy.loginfo("[DAgger] Đợi 2s để cảm biến khởi động...")
        time.sleep(2.0)

        self.trainer.start()
        rate = rospy.Rate(LOOP_RATE_HZ)

        rospy.loginfo("[DAgger] === BẮT ĐẦU VÒNG LẶP ĐIỀU KHIỂN ===")

        while not rospy.is_shutdown():
            self._loop_count += 1
            now = rospy.get_time()

            # ---- Cập nhật Perception ----
            self._update_perception()

            # ---- Build state vector S_t ----
            state_vec, info = extract_state(self.lane_state, self.latest_scan)
            lidar_zones_normalized = state_vec[3:8]  # d_left...d_right

            # ---- State Machine transitions ----
            if self.state == DAggerState.WAITING:
                if not self.e_stop_locked:
                    self._set_state(DAggerState.AI_CONTROL)

            elif self.state in (DAggerState.AI_CONTROL, DAggerState.HUMAN_CONTROL):
                if self.e_stop_locked:
                    self._set_state(DAggerState.E_STOP)
                elif self.joy_active:
                    if self.state != DAggerState.HUMAN_CONTROL:
                        self._set_state(DAggerState.HUMAN_CONTROL)
                else:
                    if self.state != DAggerState.AI_CONTROL:
                        self._set_state(DAggerState.AI_CONTROL)

            # ---- Tính lệnh điều khiển ----
            cmd_steer    = 0.0
            cmd_throttle = 0.0
            mode         = 'IDLE'
            is_safety    = False

            if self.state == DAggerState.AI_CONTROL:
                # AI dự đoán từ state vector
                ai_steer, ai_throttle = self.policy.predict(state_vec)
                cmd_steer    = ai_steer
                cmd_throttle = ai_throttle
                mode         = 'AI'

            elif self.state == DAggerState.HUMAN_CONTROL:
                # Joy override
                cmd_steer    = self.joy_steer
                cmd_throttle = self.joy_throttle
                mode         = 'JOY'

                # Thu thập dữ liệu — đây là trái tim của DAgger
                action = np.array([cmd_steer, cmd_throttle], dtype=np.float32)
                self.buffer.push(state_vec, action)
                self._joy_samples += 1

                # Auto-save định kỳ
                if self._joy_samples - self._last_autosave >= AUTOSAVE_EVERY:
                    self._last_autosave = self._joy_samples
                    self.policy.save(MODEL_PATH)
                    self.buffer.save_csv(ANCHOR_CSV)
                    rospy.loginfo(f"[DAgger] Auto-saved ({self._joy_samples} joy samples tổng cộng)")

            elif self.state == DAggerState.E_STOP:
                mode = 'ESTOP'

            # ---- Safety Layer ----
            if self.state not in (DAggerState.WAITING, DAggerState.E_STOP):
                safe_steer, safe_throttle, is_estop = self.safety.check(
                    cmd_steer, cmd_throttle, lidar_zones_normalized
                )
                if is_estop and self.state != DAggerState.SAFETY_STOP:
                    self._set_state(DAggerState.SAFETY_STOP)
                    rospy.logwarn("[DAgger] SAFETY: phanh khẩn cấp LiDAR!")
                elif not is_estop and self.state == DAggerState.SAFETY_STOP:
                    # Đường thông → trở lại
                    self._set_state(DAggerState.AI_CONTROL if not self.joy_active else DAggerState.HUMAN_CONTROL)
                cmd_steer    = safe_steer
                cmd_throttle = safe_throttle
                is_safety    = is_estop

            # ---- Gửi lệnh xuống xe ----
            if self.state in (DAggerState.E_STOP,):
                self.racer.stop()
            else:
                self.racer.steer(cmd_steer, cmd_throttle)

            # ---- Ghi log CSV ----
            self._csv.writerow([
                f"{now:.3f}", mode, self.state.name,
                f"{info['e_y_raw']:.4f}",
                f"{info['theta_e_raw']:.4f}",
                int(info['line_visible']),
                *[f"{d:.3f}" for d in info['lidar_zones_m']],
                int(info['obstacle_detected']),
                f"{cmd_steer:.4f}", f"{cmd_throttle:.4f}",
                int(is_safety),
                len(self.buffer),
                self.policy.update_count,
            ])

            # ---- Print status định kỳ ----
            if self._loop_count % PRINT_STATUS_EVERY == 0:
                rospy.loginfo(
                    f"[{self.state.name:15s}] "
                    f"steer={cmd_steer:+.2f} thr={cmd_throttle:.2f} | "
                    f"e_y={info['e_y_raw']:+.3f}m  θ={math.degrees(info['theta_e_raw']):+.1f}° | "
                    f"obst={'YES' if info['obstacle_detected'] else 'no ':3s} "
                    f"d_front={info['lidar_zones_m'][2]:.2f}m | "
                    f"buf={len(self.buffer)} joy_samples={self._joy_samples} "
                    f"updates={self.policy.update_count} loss={self.policy.last_loss:.4f}"
                )

            rate.sleep()

        self._shutdown()

    # ==================================================================
    # SHUTDOWN
    # ==================================================================

    def _shutdown(self):
        rospy.loginfo("[DAgger] === KẾT THÚC — đang dọn dẹp... ===")
        try:
            self.trainer.stop()
        except Exception:
            pass
        try:
            self.racer.stop()
        except Exception:
            pass
        try:
            self.policy.save(MODEL_PATH)
            self.buffer.save_csv(ANCHOR_CSV)
            rospy.loginfo(f"[DAgger] Đã lưu model và {len(self.buffer)} buffer samples.")
        except Exception as e:
            rospy.logerr(f"[DAgger] Lỗi khi lưu: {e}")
        try:
            self._log_file.close()
        except Exception:
            pass
        rospy.loginfo("[DAgger] Kết thúc an toàn.")

    # ==================================================================
    # HELPERS
    # ==================================================================

    @staticmethod
    def _print_controls():
        rospy.loginfo("=" * 55)
        rospy.loginfo("  GAMEPAD CONTROLS (PS4/Xbox)")
        rospy.loginfo("  Triangle (△) : Mở khoá / Bắt đầu")
        rospy.loginfo("  Circle  (○)  : E-STOP khẩn cấp")
        rospy.loginfo("  L1           : Lưu model ngay")
        rospy.loginfo("  R1           : Turbo mode")
        rospy.loginfo("  Left Stick X : Lái (khi can thiệp)")
        rospy.loginfo("  R2 / L2      : Ga / Phanh (khi can thiệp)")
        rospy.loginfo("  D-pad ↑↓     : Tăng/Giảm tốc độ cơ bản")
        rospy.loginfo("  → Nhích Joy  : TỰ ĐỘNG chuyển HUMAN + thu data")
        rospy.loginfo("  → Thả Joy    : Trả quyền về AI sau 0.5s")
        rospy.loginfo("=" * 55)


# =====================================================================
# ENTRY POINT
# =====================================================================
def main():
    rospy.init_node('dagger_controller', anonymous=True)
    controller = None
    try:
        controller = DAggerController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"[DAgger] Lỗi nghiêm trọng: {e}")
        import traceback; traceback.print_exc()
    finally:
        if controller is not None:
            try: controller._shutdown()
            except Exception: pass
        else:
            try: RacerController().stop()
            except Exception: pass


if __name__ == '__main__':
    main()
