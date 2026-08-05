# Các tham số cấu hình hệ thống

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

# Kiểu điều khiển
CONTROLLER_TYPE = 'predictive'  # 'pid' hoặc 'predictive'

# 4. ROS Topics (Khai báo chung để dễ cấu hình lại nếu đổi camera/lidar)
ROS_TOPIC_CAMERA = '/csi_cam_0/image_raw'
ROS_TOPIC_LIDAR  = '/rplidarNode'
ROS_TOPIC_JOY    = '/joy'

# 5. AI Decision Engine
AI_TURN_HOLD_TIME          = 2.5    # Giây. Thời gian giữ nguyên lệnh rẽ qua giao lộ
AI_TURN_LEFT_STEER         = -0.85  # Góc lái khi rẽ trái (-1.0 đến 1.0)
AI_TURN_RIGHT_STEER        = +0.85  # Góc lái khi rẽ phải (-1.0 đến 1.0)
AI_TURN_THROTTLE           = 0.18   # Tốc độ chậm khi đang rẽ
AI_SPEED_NORMAL            = 0.22   # Tốc độ bình thường khi bám làn
AI_INTERSECTION_MIN_FRAMES = 5      # Số frame liên tiếp mất vạch để xác nhận ngã tư
AI_TURN_PRIORITY           = ['left', 'right', 'straight']  # Ưu tiên rẽ mặc định

# 6. Traffic Detector (Nhận diện đèn giao thông & biển báo)
TRAFFIC_ROI_TOP_RATIO   = 0.45   # Tỉ lệ chiều cao vùng ROI (quét nửa trên ảnh)
TRAFFIC_MIN_AREA_LIGHT  = 80     # Pixel^2. Diện tích tối thiểu để coi là đèn thật
TRAFFIC_MIN_AREA_SIGN   = 150    # Pixel^2. Diện tích tối thiểu để coi là biển thật
TRAFFIC_HISTORY_LEN     = 5      # Số frame dùng để ổn định kết quả (Voting)
# Dải màu HSV cho đèn ĐỎ (2 dải vì Đỏ nằm ở 2 đầu vòng HSV)
TRAFFIC_RED_HSV_LOWER1  = [0,   120, 100]
TRAFFIC_RED_HSV_UPPER1  = [10,  255, 255]
TRAFFIC_RED_HSV_LOWER2  = [160, 120, 100]
TRAFFIC_RED_HSV_UPPER2  = [180, 255, 255]
# Dải màu HSV cho đèn XANH LÁ
TRAFFIC_GREEN_HSV_LOWER = [40,  80,  80]
TRAFFIC_GREEN_HSV_UPPER = [85,  255, 255]
# Dải màu HSV cho biển báo nền XANH DƯƠNG
TRAFFIC_BLUE_HSV_LOWER  = [100, 100, 80]
TRAFFIC_BLUE_HSV_UPPER  = [130, 255, 255]
