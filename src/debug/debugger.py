import csv
import time
import os
import math
import cv2
import numpy as np


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
        self.cam_writer = None
        self.lidar_writer = None
        self.info_path = None

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

            # Khởi tạo file CSV (thêm cột throttle và ai_action so với bản cũ)
            csv_path = os.path.join(session_dir, "speed_track_debug.csv")
            self.csv_file = open(csv_path, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'timestamp', 'state', 'front_dist', 'closest_angle',
                'closest_dist', 'current_offset_px', 'steering',
                'throttle', 'ai_action'
            ])

            # Khởi tạo VideoWriter
            # Dung 'XVID' + duoi '.avi' - codec on dinh nhat, khong can thu vien ngoai
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            cam_vid_path   = os.path.join(session_dir, "camera_log.avi")
            lidar_vid_path = os.path.join(session_dir, "lidar_log.avi")
            # Kich thuoc co dinh 300x300, toc do 20fps
            self.cam_writer   = cv2.VideoWriter(cam_vid_path,   fourcc, 20.0, (300, 300))
            self.lidar_writer = cv2.VideoWriter(lidar_vid_path, fourcc, 20.0, (300, 300))

            self._dbg(f"[Debugger] Session {session_num} bat dau. Log tai: {session_dir}")

    # ----------------------------------------------------------------
    # Helper: in log an toan (rospy.logdebug hoac print fallback)
    # ----------------------------------------------------------------
    def _dbg(self, message):
        """In mot dong debug. Dung rospy.logdebug neu co ROS, nguoc lai dung print."""
        if not self.debug_mode:
            return
        if self._has_rospy:
            self._rospy.logdebug(message)
        else:
            print(f"[DEBUG] {message}")

    def _info(self, message):
        """In log tong ket chu ky (luon hien, throttle 0.5s neu co ROS)."""
        if self._has_rospy:
            self._rospy.loginfo_throttle(0.5, message)
        else:
            print(f"[INFO]  {message}")

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
                round(time.time(), 3), state,
                round(front_dist, 3),
                round(closest_angle, 2),
                round(closest_dist, 3),
                round(offset_px, 1),
                round(steering, 3),
                round(throttle, 3),
                ai_action
            ])

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
        if self.cam_writer:
            self.cam_writer.release()
        if self.lidar_writer:
            self.lidar_writer.release()

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
            state, front_dist, closest_angle, front_dist,
            offset_px, steering,
            throttle=throttle,
            ai_action=ai_action or ''
        )

        # ---- Ghi video Camera ----
        latest_image = blackboard.get('latest_image')
        if latest_image is not None and self.cam_writer:
            display_img = latest_image.copy()

            # Ve waypoints mau do
            for pt in waypoints:
                cv2.circle(display_img, pt, 5, (0, 0, 255), -1)

            # Ve duong cong du doan mau xanh la
            predicted_curve = blackboard.get('predicted_curve', [])
            if len(predicted_curve) >= 2:
                pts = np.array(predicted_curve, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_img, [pts], False, (0, 255, 0), 2)

            # Chu thich trang thai len anh
            label = state if not ai_action else f"{state} | {ai_action}"
            cv2.putText(display_img, label,
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(display_img, f"Dist:{front_dist:.2f}m  Steer:{steering:+.3f}",
                        (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 0), 1)

            if display_img.shape[:2] != (300, 300):
                display_img = cv2.resize(display_img, (300, 300))
            self.cam_writer.write(display_img)
            self._dbg("[Debugger | Video] Ghi frame Camera OK.")

        # ---- Ghi video Lidar ----
        latest_scan = blackboard.get('latest_scan')
        if latest_scan is not None and self.lidar_writer:
            lidar_img = self.visualize_lidar(latest_scan)
            self.lidar_writer.write(lidar_img)
            self._dbg("[Debugger | Video] Ghi frame Lidar OK.")
