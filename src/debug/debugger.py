import csv
import sys
import time
import os
import math
import traceback
import cv2
import numpy as np
from src.config import settings


class Debugger:
    """
    Lớp tiện ích debug tập trung cho toàn bộ hệ thống robot.

    Đây là nơi DUY NHẤT thực hiện:  
      - In log chi tiết (rospy.logdebug) mỗi chu kỳ xử lý
      - Ghi log dữ liệu ra file CSV
      - Ghi video Camera và Lidar

    Thiết kế an toàn: Có thể dùng cả khi có ROS và khi không có ROS
    (ví dụ: trong unit test). Khi không có ROS, log sẽ được in qua print().
    """

    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.csv_file = None
        self.csv_writer = None
        self.combined_writer = None
        self.raw_writer = None      # Video thô chưa qua xử lí
        self.canny_writer = None    # Video viền Canny chưa qua xử lí
        self.debug_log_file = None  # File log dành riêng cho [DEBUG]
        self.info_path = None
        self.session_dir = None
        
        # Bien de tinh FPS
        self._last_time = 0.0
        self._fps = 0.0

        # Thử import rospy một lần - dùng cho toàn bộ vòng đời đối tượng
        self._rospy = None
        self._has_rospy = False
        try:
            import rospy
            self._rospy = rospy
            self._has_rospy = True
        except ImportError:
            pass

        if self.debug_mode:
            import glob
            import datetime
            # Đọc kích thước ảnh từ settings
            try:
                import sys as _sys
                _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from config import settings as _cfg
                _img_w = getattr(_cfg, 'IMAGE_WIDTH', 640)
                _img_h = getattr(_cfg, 'IMAGE_HEIGHT', 480)
            except Exception:
                _img_w, _img_h = 640, 480

            # Khởi tạo thư mục log cơ sở
            repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_base_dir = os.path.join(repo_dir, "logs")
            if not os.path.exists(log_base_dir):
                os.makedirs(log_base_dir)

            # Xác định session tiếp theo
            existing_sessions = glob.glob(os.path.join(log_base_dir, "session_*"))
            max_session = 0
            for s in existing_sessions:
                try:
                    num = int(os.path.basename(s).split("_")[1])
                    if num > max_session:
                        max_session = num
                except Exception:
                    pass
            session_num = max_session + 1
            session_dir = os.path.join(log_base_dir, f"session_{session_num}")
            os.makedirs(session_dir)

            # Khởi tạo file ghi chú thông tin
            self.info_path = os.path.join(session_dir, "session_info.txt")
            start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.info_path, 'w', encoding='utf-8') as f:
                f.write(f"Session {session_num}\n")
                f.write(f"Bat dau: {start_time_str}\n")

            # Khởi tạo file log cho [DEBUG]
            debug_log_path = os.path.join(session_dir, "debug.log")
            self.debug_log_file = open(debug_log_path, mode='w', encoding='utf-8')

            # Khởi tạo file CSV (thêm cột throttle và ai_action so với bản cũ)
            csv_path = os.path.join(session_dir, "speed_track_debug.csv")
            self.csv_file = open(csv_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'timestamp', 'state', 'front_dist', 'closest_angle',
                'closest_dist', 'current_offset_px', 'steering',
                'throttle', 'ai_action'
            ])

            # Khởi tạo VideoWriter cho video debug đã xử lý (combined)
            # 4 panels 300x300: [Camera | BEV | Threshold | Lidar] = 1200x300
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            combined_vid_path = os.path.join(session_dir, "combined_log.avi")
            self.combined_writer = cv2.VideoWriter(combined_vid_path, fourcc, 20.0, (1200, 300))

            # Khởi tạo VideoWriter cho video thô chưa qua xử lý (raw camera)
            raw_vid_path = os.path.join(session_dir, "raw_camera.avi")
            self.raw_writer = cv2.VideoWriter(raw_vid_path, fourcc, 20.0, (_img_w, _img_h))
            self.raw_width = _img_w
            self.raw_height = _img_h

            # Khởi tạo VideoWriter cho viền Canny sạch
            canny_vid_path = os.path.join(session_dir, "canny_edges.avi")
            self.canny_writer = cv2.VideoWriter(canny_vid_path, fourcc, 20.0, (_img_w, _img_h))

 
            # Lưu session_dir để dùng sau (vd: ghi crash log)
            self.session_dir = session_dir

            self._dbg(f"[Debugger] Session {session_num} bat dau. Log tai: {session_dir}")

    # ----------------------------------------------------------------
    # Helper: in log an toan (rospy.logdebug hoac print fallback)
    # ----------------------------------------------------------------
    def _dbg(self, message):
        """Ghi log debug ra file debug.log và rospy.logdebug, không in ra console để tránh loãng màn hình."""
        if not self.debug_mode:
            return
        # Ghi vào file debug.log
        if self.debug_log_file:
            try:
                import datetime
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.debug_log_file.write(f"[{ts}] [DEBUG] {message}\n")
                self.debug_log_file.flush()
            except Exception:
                pass
        if self._has_rospy:
            self._rospy.logdebug(message)

    def _info(self, message):
        """In log tong ket chu ky (rospy loginfo_throttle va print ra console)."""
        # Giới hạn tần suất in ra console 0.5s giống rospy để tránh tràn màn hình
        curr_time = time.time()
        if not hasattr(self, '_last_print_time'):
            self._last_print_time = 0.0
        if curr_time - self._last_print_time >= 0.5:
            print(f"[INFO]  {message}", flush=True)
            self._last_print_time = curr_time

        if self._has_rospy:
            self._rospy.loginfo_throttle(0.5, message)

    def log_error(self, exc: BaseException, context: str = ""):
        """
        In lỗi đầy đủ (traceback) ra console (stderr) và vào ROS log.
        Nên gọi trong khối except khi xảy ra lỗi bên trong vòng lặp chính.
        """
        tb_str = traceback.format_exc()
        label = f"[Debugger ERROR] {context + ' | ' if context else ''}{type(exc).__name__}: {exc}"

        # In ra console stderr ngay lập tức
        print(label, file=sys.stderr, flush=True)
        print(tb_str, file=sys.stderr, flush=True)

        # Ghi qua rospy nếu có
        if self._has_rospy:
            self._rospy.logerr(label)
            self._rospy.logerr(tb_str)

        # Ghi vào file error.log trong session_dir
        if self.session_dir:
            try:
                import datetime
                err_path = os.path.join(self.session_dir, "error.log")
                with open(err_path, "a", encoding="utf-8") as f:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n[{ts}] {label}\n{tb_str}\n")
            except Exception:
                pass

    # ----------------------------------------------------------------
    # API cong khai
    # ----------------------------------------------------------------
    def log(self, message):
        """In log ra man hinh neu dang o che do debug."""
        if self.debug_mode:
            self._dbg(message)

    def log_csv(self, state, front_dist, closest_angle, closest_dist,
                offset_px, steering, throttle=0.0, ai_action=''):
        """Luu tru du lieu vao CSV sau moi chu ky."""
        if self.debug_mode and self.csv_writer:
            self.csv_writer.writerow([
                round(time.time(), 3), str(state),
                round(front_dist, 3),
                round(closest_angle, 2),
                round(closest_dist, 3),
                round(offset_px, 1),
                round(steering, 3),
                round(throttle, 3),
                str(ai_action) if ai_action else ''
            ])
            self.csv_file.flush() # Bắt buộc ghi file ngay lập tức thay vì đưa vào bộ đệm

    def show_image(self, window_name, image):
        """Hien thi anh bang OpenCV neu dang o che do debug."""
        if self.debug_mode:
            try:
                cv2.imshow(window_name, image)
                cv2.waitKey(1)
            except Exception as e:
                print(f"[WARNING] Khong the hien thi anh: {e}")

    def visualize_lidar(self, scan_data):
        """Ve du lieu Lidar ra mot buc anh den 300x300."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cx, cy = 150, 150
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)

        # 30 pixels = 1 met
        scale = 30.0
        for angle_deg, dist in scan_data:
            angle_rad = math.radians(angle_deg - 90)
            x = int(cx + dist * scale * math.cos(angle_rad))
            y = int(cy + dist * scale * math.sin(angle_rad))
            if 0 <= x < 300 and 0 <= y < 300:
                cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

        # Luoi toa do (vong tron ban kinh 1m, 2m, ...)
        for r in range(1, 6):
            cv2.circle(img, (cx, cy), int(r * scale), (50, 50, 50), 1)

        return img

    def close(self):
        """Don dep tai nguyen khi tat."""
        if self.csv_file:
            self.csv_file.close()
        if self.debug_log_file:
            self.debug_log_file.close()
        if self.combined_writer:
            self.combined_writer.release()
        if self.raw_writer:
            self.raw_writer.release()
        if self.canny_writer:
            self.canny_writer.release()

        if hasattr(self, 'info_path') and self.info_path:
            import datetime
            end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.info_path, 'a', encoding='utf-8') as f:
                f.write(f"Ket thuc: {end_time_str}\n")

    # ----------------------------------------------------------------
    # process() - diem DUY NHAT nhan Blackboard va xuat toan bo debug
    # ----------------------------------------------------------------
    def process(self, blackboard):
        """
        Ham chuan giao tiep Blackboard.
        Day la noi DUY NHAT in toan bo thong tin debug ra terminal
        (rospy.logdebug) va ghi vao file (CSV, video).

        Tuong thich voi:
          - ros_speed_track.py  (khong co AI)
          - ros_ai_navigation.py (co AI)
          - test_blackboard_flow.py (khong co ROS)
        """
        if not self.debug_mode:
            return

        # ---- Doc du lieu tu Blackboard ----
        state         = blackboard.get('state_name', 'UNKNOWN')
        front_dist    = blackboard.get('front_dist', 999.0)
        closest_angle = blackboard.get('closest_angle', 0.0)
        closest_dist  = blackboard.get('closest_dist', 999.0)
        offset_px     = blackboard.get('current_offset_px', 0.0)
        steering      = blackboard.get('steering', 0.0)
        throttle      = blackboard.get('throttle', 0.0)
        center_x      = blackboard.get('center_x', 0.0)
        waypoints     = blackboard.get('lane_waypoints', [])
        side_clear    = blackboard.get('side_clear', True)
        dodge_dir     = blackboard.get('dodge_direction', 0.0)

        # Du lieu AI (None neu task khong dung AI)
        ai_action   = blackboard.get('ai_action',  None)
        ai_steer    = blackboard.get('ai_steering', None)
        ai_throttle = blackboard.get('ai_throttle', None)

        # ---- rospy.logdebug: Chi tiet tung module ----
        # (Chi thay khi bat: python3 <task>.py __log:=debug)

        self._dbg(
            f"[Debugger | FSM  ] State={state:12s} | "
            f"Dodge={dodge_dir:+.1f} | Offset={offset_px:+.1f}px | Side_Clear={side_clear}"
        )
        self._dbg(
            f"[Debugger | Lidar] Front={front_dist:.3f}m | "
            f"Angle={closest_angle:+.1f}deg | Side_Clear={side_clear}"
        )
        self._dbg(
            f"[Debugger | Cam  ] Center_X={center_x:.1f}px | "
            f"Waypoints={len(waypoints)} | Offset={offset_px:+.1f}px"
        )
        self._dbg(
            f"[Debugger | Ctrl ] Steering={steering:+.4f} | Throttle={throttle:+.3f}"
        )
        if ai_action is not None:
            self._dbg(
                f"[Debugger | AI   ] Action={ai_action:15s} | "
                f"Steer={ai_steer:+.4f} | Throttle={ai_throttle:+.3f}"
            )

        # ---- Tong ket chu ky (luon hien, moi 0.5s) ----
        if ai_action is not None:
            self._info(
                f"[CYCLE] FSM={state} | AI={ai_action:15s} | "
                f"Steer={ai_steer:+.3f} | Thr={ai_throttle:+.2f} | "
                f"Dist={front_dist:.2f}m | WPs={len(waypoints)}"
            )
        else:
            self._info(
                f"[CYCLE] FSM={state} | "
                f"Steer={steering:+.3f} | Thr={throttle:+.2f} | "
                f"Dist={front_dist:.2f}m | Offset={offset_px:+.1f}px | WPs={len(waypoints)}"
            )

        # ---- Ghi CSV ----
        self.log_csv(
            state, front_dist, closest_angle, closest_dist,
            offset_px, steering,
            throttle=throttle,
            ai_action=ai_action
        )

        # ---- Tinh FPS ----
        curr_time = time.time()
        if self._last_time > 0:
            dt = curr_time - self._last_time
            fps_current = 1.0 / dt if dt > 0 else 0.0
            self._fps = 0.8 * self._fps + 0.2 * fps_current
        self._last_time = curr_time

        # ---- Hợp nhất 4 ảnh vào 1 frame (1200x300): [Camera | BEV | Threshold | Lidar] ----
        combined_img = np.zeros((300, 1200, 3), dtype=np.uint8)

        # 1. Camera goc
        latest_image = blackboard.get('latest_image')
        if latest_image is not None:
            display_img = latest_image.copy()
            h, w = display_img.shape[:2]
            
            # Vẽ đường tâm tuyệt đối của camera (Màu xám)
            cv2.line(display_img, (w // 2, 0), (w // 2, h), (100, 100, 100), 1)
            
            # Vẽ vùng an toàn (Safe zone)
            safe_margin = int(settings.SAFE_ZONE_PERCENT * (w / 2.0))
            left_safe = w // 2 - safe_margin
            right_safe = w // 2 + safe_margin
            # Vẽ 2 vạch giới hạn vùng an toàn (Màu xanh lá cây nhạt)
            cv2.line(display_img, (left_safe, 0), (left_safe, h), (0, 100, 0), 1)
            cv2.line(display_img, (right_safe, 0), (right_safe, h), (0, 100, 0), 1)
            
            # Vẽ raw waypoints chưa lọc EMA (Màu cam)
            raw_wps = blackboard.get('raw_waypoints', [])
            for pt in raw_wps:
                cv2.circle(display_img, pt, 3, (0, 165, 255), -1)
            
            # Ve waypoints đã qua EMA va duong noi (Mau do, cham vang)
            if len(waypoints) > 0:
                pts = np.array(waypoints, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_img, [pts], False, (0, 0, 255), 1) # Đường nối đỏ mảnh
                for pt in waypoints:
                    cv2.circle(display_img, pt, 5, (0, 255, 255), -1) # Điểm EMA vàng
            
            # Vẽ điểm điều khiển Lookahead (Màu xanh ngọc cyan)
            lookahead_pt = blackboard.get('lookahead_point')
            if lookahead_pt is not None:
                cv2.circle(display_img, lookahead_pt, 7, (255, 255, 0), 2)
                cx, cy = lookahead_pt
                cv2.line(display_img, (cx - 10, cy), (cx + 10, cy), (255, 255, 0), 1)
                cv2.line(display_img, (cx, cy - 10), (cx, cy + 10), (255, 255, 0), 1)
                    
            predicted_curve = blackboard.get('predicted_curve', [])
            if len(predicted_curve) >= 2:
                pts = np.array(predicted_curve, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_img, [pts], False, (0, 255, 0), 2)
            
            label = state if not ai_action else f"{state} | {ai_action}"
            cv2.putText(display_img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(display_img, f"Dist:{front_dist:.2f}m Steer:{steering:+.2f} Thr:{throttle:.2f}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 0), 1)
            
            if display_img.shape[:2] != (300, 300):
                display_img = cv2.resize(display_img, (300, 300))
            combined_img[:, 0:300] = display_img

        # 2. BEV Debug (chỉ có khi USE_BOUNDARY_PATH=True)
        bev_debug_img = blackboard.get('bev_debug_img')
        if bev_debug_img is not None:
            bev_panel = bev_debug_img.copy()
            if bev_panel.shape[:2] != (300, 300):
                bev_panel = cv2.resize(bev_panel, (300, 300))
            # Thêm nhãn (dời xuống y=40 để tránh đè lên text màu vàng)
            cv2.putText(bev_panel, "BEV + Boundary Fit", (5, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
        else:
            # Placeholder: Hiển thị nền đen + chú thích khi không dùng BEV
            bev_panel = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.putText(bev_panel, "BEV (disabled)", (60, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        combined_img[:, 300:600] = bev_panel

        # 3. Camera Thresh
        camera_thresh = blackboard.get('camera_thresh')
        if camera_thresh is not None:
            thresh_resized = cv2.resize(camera_thresh, (300, 300))
            if thresh_resized.ndim == 3 and thresh_resized.shape[2] == 3:
                thresh_color = thresh_resized
            else:
                thresh_color = cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2BGR)
            
            # Ve waypoint len Threshold de kiem tra su an khop truc quan nhat
            if len(waypoints) > 0:
                pts = np.array(waypoints, np.int32).reshape((-1, 1, 2))
                cv2.polylines(thresh_color, [pts], False, (0, 0, 255), 2)
                for pt in waypoints:
                    cv2.circle(thresh_color, pt, 4, (0, 255, 255), -1)
            
            # Them chu thich (dời xuống y=40 để tránh đè lên text màu vàng)
            cv2.putText(thresh_color, "Threshold (Binary)", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            combined_img[:, 600:900] = thresh_color

        # 4. Lidar
        latest_scan = blackboard.get('latest_scan')
        if latest_scan is not None:
            lidar_img = self.visualize_lidar(latest_scan)
            cv2.putText(lidar_img, "Lidar Map", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            combined_img[:, 900:1200] = lidar_img

        # ---- Ve FPS chung ----
        cv2.putText(combined_img, f"FPS: {self._fps:.1f}", (1100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ---- Ghi video và hiển thị ----
        # Ghi raw frame - ưu tiên dùng raw_camera_frame (frame gốc chưa vẽ overlay)
        raw_img = blackboard.get('raw_camera_frame') if blackboard.get('raw_camera_frame') is not None else blackboard.get('latest_image')
        if raw_img is not None and self.raw_writer:
            try:
                raw_bgr = raw_img if raw_img.ndim == 3 else cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
                raw_resized = cv2.resize(raw_bgr, (self.raw_width, self.raw_height))
                self.raw_writer.write(raw_resized)
            except Exception as e:
                self._dbg(f"[Debugger | RawVideo] Lỗi ghi raw frame: {e}")


        # Ghi canny frame sạch
        canny_img = blackboard.get('canny_edges')
        if canny_img is not None and self.canny_writer:
            try:
                canny_bgr = canny_img if canny_img.ndim == 3 else cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
                canny_resized = cv2.resize(canny_bgr, (self.raw_width, self.raw_height))
                self.canny_writer.write(canny_resized)
            except Exception as e:
                self._dbg(f"[Debugger | CannyVideo] Lỗi ghi canny frame: {e}")

        if self.combined_writer:
            self.combined_writer.write(combined_img)
            self._dbg("[Debugger | Video] Ghi combined frame OK.")
        
        # self.show_image("JetRacer Combined Debug", combined_img)

