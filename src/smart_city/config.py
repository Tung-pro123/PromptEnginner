#!/usr/bin/env python3
"""
Smart City v2 — Centralized Configuration

Tất cả tham số tune được ở một nơi. Không magic numbers trong code.
Tham số [CALIBRATE] cần đo/chỉnh trên xe thật.
Tham số [TUNE] cần tinh chỉnh sau khi test trên sa bàn.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SmartCityConfig:
    """Config cho Smart City v2 pipeline."""

    # ================================================================
    # CAMERA / IMAGE
    # ================================================================
    image_width: int = 300
    image_height: int = 300

    # ================================================================
    # ROI — Dual Lane Detection
    # ================================================================
    # ROI thực thi (gần xe) — dùng để bám line chính xác
    roi_y_ratio: float = 0.80           # [TUNE] bắt đầu từ 80% chiều cao
    roi_h_ratio: float = 0.15           # [TUNE] chiều cao ROI = 15%

    # ROI dự báo (xa phía trước) — phát hiện sớm mất line / giao lộ
    lookahead_y_ratio: float = 0.55     # [TUNE]
    lookahead_h_ratio: float = 0.15     # [TUNE]

    # Focus mask: chỉ giữ vùng giữa ảnh (loại nhiễu rìa)
    focus_width_percent: float = 0.85   # [TUNE] 85% giữa ảnh

    # ================================================================
    # HSV COLOR SEGMENTATION — Line biên
    # ================================================================
    # Mặc định: phát hiện line ĐEN trên nền sáng hơn (giống code smart city cũ)
    # LINE_COLOR_LOWER/UPPER trong HSV
    line_hsv_lower: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0])
    )
    line_hsv_upper: np.ndarray = field(
        default_factory=lambda: np.array([180, 255, 75])
    )

    # Morphology
    morph_kernel_size: int = 3
    morph_open_iter: int = 1
    morph_close_iter: int = 2

    # Contour
    min_contour_area: int = 80          # [TUNE] diện tích tối thiểu contour

    # ================================================================
    # LANE GEOMETRY
    # ================================================================
    # Khoảng cách pixel ước lượng giữa 2 line biên trong ROI
    # [CALIBRATE] Đo trên sa bàn thật: chạy debug, xem 2 biên cách nhau bao nhiêu pixel
    expected_lane_width_px: int = 140   # [CALIBRATE] ~140px trên ảnh 300x300

    # Khi chỉ thấy 1 biên, offset bao nhiêu pixel để ước lượng center
    lane_half_width_px: int = 70        # = expected_lane_width_px / 2

    # ================================================================
    # CROSSWALK DETECTION
    # ================================================================
    # ROI cho crosswalk (phía trước xa hơn ROI dự báo)
    crosswalk_y_ratio: float = 0.40     # [TUNE]
    crosswalk_h_ratio: float = 0.15     # [TUNE]

    # Ngưỡng: 1 hàng ngang có > X% pixel trắng → coi là "vạch ngang"
    crosswalk_row_fill_ratio: float = 0.40  # [TUNE] >40% chiều ngang

    # Cần ít nhất N hàng ngang thỏa → crosswalk detected
    crosswalk_min_rows: int = 8         # [TUNE]

    # Cooldown: sau khi phát hiện crosswalk, bỏ qua N giây
    crosswalk_cooldown_sec: float = 3.0

    # ================================================================
    # TEMPORAL FILTER (EMA)
    # ================================================================
    # Làm mượt centerline ảo qua các frame
    ema_alpha: float = 0.6              # [TUNE] 0=giữ cũ, 1=lấy mới hoàn toàn

    # ================================================================
    # PID STEERING
    # ================================================================
    pid_kp: float = 0.015               # [TUNE]
    pid_ki: float = 0.0                 # [TUNE]
    pid_kd: float = 0.003               # [TUNE]

    # ================================================================
    # SPEED CONTROL
    # ================================================================
    base_speed: float = 0.16            # [TUNE] tốc độ đi thẳng
    curve_speed: float = 0.12           # [TUNE] tốc độ khi cua (error lớn)
    intersection_speed: float = 0.14    # [TUNE] tốc độ vào giao lộ
    recover_speed: float = 0.10         # [TUNE] tốc độ tìm line

    # Ngưỡng error pixel để giảm tốc khi cua
    curve_error_thresh: float = 35      # [TUNE] pixel

    # ================================================================
    # STEERING
    # ================================================================
    max_correction: float = 0.15        # [TUNE] giới hạn góc lái tối đa
    steer_invert: bool = True           # [CALIBRATE] True nếu servo bị ngược

    # ================================================================
    # INTERSECTION HANDLING
    # ================================================================
    intersection_approach_duration: float = 0.5     # [TUNE] giây đi thẳng vào ngã
    intersection_clearance_duration: float = 1.5    # [TUNE] giây đi thẳng ra ngã
    line_reacquire_timeout: float = 3.0             # [TUNE] timeout tìm lại line

    # ================================================================
    # TURN
    # ================================================================
    turn_speed: float = 0.2             # [TUNE]
    turn_duration_90_deg: float = 0.8   # [CALIBRATE] giây cho 90 độ

    # ================================================================
    # TIMING
    # ================================================================
    loop_rate: int = 20                 # Hz
    wait_timeout: float = 30.0          # giây chờ tìm line ban đầu

    # ================================================================
    # VIDEO / DEBUG
    # ================================================================
    record_video: bool = True
    video_fps: int = 20

    # ================================================================
    # ROS TOPICS
    # ================================================================
    camera_topic: str = '/csi_cam_0/image_raw'
    lidar_topic: str = '/scan'

    # ================================================================
    # ROBOFLOW AI DETECTION (biển báo + đèn giao thông)
    # ================================================================
    rf_model: str = "dataset3-c4kyj"
    rf_version: str = "1"
    # API key lấy từ env var ROBOFLOW_API_KEY
    rf_conf_threshold: float = 0.6

    # ================================================================
    # MAP / NAVIGATION
    # ================================================================
    # Đường dẫn mặc định đến file map.json
    # Sẽ được resolve relative trong main
    map_filename: str = "map.json"

    # ================================================================
    # MQTT
    # ================================================================
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_data_topic: str = "jetbot/corrected_event_data"

    # ================================================================
    # COMPUTED PROPERTIES (tự tính từ các giá trị trên)
    # ================================================================
    @property
    def roi_y(self) -> int:
        return int(self.image_height * self.roi_y_ratio)

    @property
    def roi_h(self) -> int:
        return int(self.image_height * self.roi_h_ratio)

    @property
    def lookahead_y(self) -> int:
        return int(self.image_height * self.lookahead_y_ratio)

    @property
    def lookahead_h(self) -> int:
        return int(self.image_height * self.lookahead_h_ratio)

    @property
    def crosswalk_y(self) -> int:
        return int(self.image_height * self.crosswalk_y_ratio)

    @property
    def crosswalk_h(self) -> int:
        return int(self.image_height * self.crosswalk_h_ratio)
