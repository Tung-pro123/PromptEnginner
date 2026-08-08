# -*- coding: utf-8 -*-
"""
Tập trung quản lý tham số cấu hình hệ thống Speed Track & JetRacer Pro.
"""

# =============================================================================
# 1. CAMERA & LANE DETECTION (XA HÌNH MỚI: VẠCH ĐỎ + NỀN TRẮNG + LÒNG ĐƯỜNG ĐEN)
# =============================================================================
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IMAGE_CENTER_X = 150

# Vùng quét ROI (Near & Far)
ROI_Y_NEAR_RATIO = 0.85  # Dòng quét gần xe (85% chiều cao)
ROI_Y_FAR_RATIO = 0.55   # Dòng quét xa xe (55% chiều cao)

# Ngưỡng màu HSV cho vạch biên ĐỎ & vạch đứt ĐỎ
HSV_RED_LOWER1 = [0, 70, 70]
HSV_RED_UPPER1 = [10, 255, 255]
HSV_RED_LOWER2 = [160, 70, 70]
HSV_RED_UPPER2 = [180, 255, 255]

# Ngưỡng màu xám cho nền TRẮNG xung quanh đường đen
WHITE_BACKGROUND_THRESHOLD = 170

# Ước lượng chiều rộng làn đường mặc định (pixel)
DEFAULT_LANE_WIDTH = 240.0
MAX_GAP_BETWEEN_POINTS = 15

# =============================================================================
# 2. LIDAR & FSM NÉ VẬT CẢN (DODGING & REENTERING)
# =============================================================================
TRIGGER_DIST = 0.70          # Mét. Nhỏ hơn cự ly này sẽ kích hoạt né vật cản
FRONT_ANGLE_MIN = -35.0      # Độ. Góc quét hình nêm trước mặt (trái)
FRONT_ANGLE_MAX = 35.0       # Độ. Góc quét hình nêm trước mặt (phải)

DODGE_OFFSET_PX = 70.0       # Pixel. Độ lệch vạch ảo khi né tránh
RAMP_STEP_DODGE_PX = 5.0     # Pixel/frame. Tốc độ dịch vạch mượt khi né gấp
RAMP_STEP_RETURN_PX = 2.0    # Pixel/frame. Tốc độ dịch vạch thoải mượt khi trả làn

SIDE_CLEAR_DIST = 0.80       # Mét. Cự ly an toàn hông xe
CLEAR_FRAMES_REQUIRED = 8    # Số frame liên tiếp an toàn hông xe để xác nhận qua hẳn vật cản
WATCHDOG_TIMEOUT = 3.5       # Giây. Thời gian né tối đa trước khi ngắt an toàn

# Phân vùng nhận diện vạch đơn (Zone-Based Prior)
ZONE_LEFT_MAX = 110          # x < 110px: Chắc chắn là biên trái
ZONE_RIGHT_MIN = 190         # x > 190px: Chắc chắn là biên phải

# Safety Steering Override trong trạng thái Né
MIN_DODGE_STEERING = 0.28    # Khóa góc lái tối thiểu khi né tránh để chống nhiễu camera

# Giao thức Trả làn 2 Giai đoạn (Two-Stage Re-entering)
OPEN_LOOP_RETURN_TIME = 1.2  # Giây. Thời gian ép lái mở quay đầu xe về lòng đường
OPEN_LOOP_STEER_ANGLE = 0.50 # Góc bẻ lái cố định trong giai đoạn 1 ép lái mở
MIN_REENTERING_DURATION = 2.5# Giây. Thời gian tối thiểu ở trạng thái 3 để bám làn ổn định

# =============================================================================
# 3. CONTROL & DRIVE CONFIG
# =============================================================================
CONTROLLER_TYPE = 'p_control'  # Option: 'p_control' hoặc 'lqr'
BASE_SPEED = 0.40              # Tốc độ ga cơ bản của xe (0.0 -> 1.0)
KP = 0.007                     # Hệ số nhạy bẻ lái P-Controller
MAX_STEERING = 1.0             # Giới hạn vật lý lái servo trái/phải [-1.0, 1.0]
MIN_STEERING = -1.0

# Tham số LQR Controller (Kinematic Bicycle Model)
LQR_WHEELBASE = 0.18           # Chiều dài cơ sở xe (mét)
LQR_SCALE_FACTOR = 0.0015      # Quy đổi 1px ~ 1.5mm
LQR_Q_DIAG = [15.0, 1.0, 8.0, 0.5] # Phạt sai số khoảng cách e & sai số góc e_theta
LQR_R_VAL = [[1.2]]

# =============================================================================
# 4. ROS TOPICS & DEBUG
# =============================================================================
ROS_TOPIC_CAMERA = '/csi_cam_0/image_raw'
ROS_TOPIC_LIDAR = '/scan'
VIDEO_OUTPUT_FILENAME = 'speed_track_run.avi'
CSV_DEBUG_FILENAME = 'speed_track_debug.csv'
