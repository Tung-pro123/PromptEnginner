# Các tham số cấu hình hệ thống (Speed Track)

# 1. Camera & Lane Detection
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IMAGE_CENTER_X = 150
THRESHOLD_VALUE = 180
MAX_GAP_BETWEEN_POINTS = 15

# 2. Control (PID Controller)
PID_KP = 0.5
PID_KI = 0.0
PID_KD = 0.1
BASE_SPEED = 0.20        # Tốc độ cơ bản
MAX_THROTTLE = 0.40      # Giới hạn tốc độ
MAX_STEERING = 1.0       # Giới hạn góc lái
MIN_STEERING = -1.0
STEERING_OFFSET = 0.0    # Bù lệch góc lái
SAFE_ZONE_PERCENT = 0.3  # Vùng an toàn không cần đánh lái

# 3. Lidar & FSM Dodging
TRIGGER_DIST = 0.70          # Mét. Nhỏ hơn khoảng cách này sẽ kích hoạt né
FRONT_ANGLE_RANGE = 35       # Độ. Vùng quét phía trước (+-35 độ)
SIDE_ANGLE_CLEAR = 110       # Độ. Vượt quá góc này ở sườn xe coi như an toàn
SIDE_CLEAR_DIST = 0.3        # Mét. Khoảng cách an toàn sườn xe
WATCHDOG_TIMEOUT = 3.5       # Giây. Thời gian tối đa né vật cản
DODGE_OFFSET_PX = 70.0       # Pixel. Khoảng cách dịch vạch ảo khi né
OFFSET_STEP = 5.0            # Pixel/frame. Độ mượt khi dịch vạch
CLEAR_FRAMES_REQUIRED = 8    # Số frame liên tiếp cần để xác nhận an toàn sườn xe

# Điều khiển
CONTROLLER_TYPE = 'predictive'  # 'pid' hoặc 'predictive'
