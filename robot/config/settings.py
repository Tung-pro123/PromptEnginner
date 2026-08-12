# Các tham số cấu hình hệ thống

# 1. Camera & Lane Detection
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IMAGE_CENTER_X = 150
THRESHOLD_VALUE = 180
MAX_GAP_BETWEEN_POINTS = 15
USE_ADVANCED_SEGMENTATION = True  # True: Dùng phân đoạn ảnh nâng cao (Sliding Window), False: Dùng quét quét hàng đơn giản
USE_COLOR_SEGMENTATION = True     # True: Dùng lọc màu HSV + Sliding Window, False: Dùng Grayscale + Sliding Window
USE_BOUNDARY_PATH = True          # True: Dùng thuật toán Boundary Following mới (detect_boundary_path)

# 1b. Boundary Path Following (detect_boundary_path)
BOUNDARY_OFFSET_PX = 55   # Pixel. Khoảng cách offset vào trong từ vạch biên đến quỹ đạo xe (tăng = xa biên hơn)
# Bird's Eye View Perspective Transform - 4 điểm nguồn (camera space)
# Thứ tự: [top-left, top-right, bottom-right, bottom-left]
# Cần calibrate lại theo góc camera thực tế trên xe.
BEV_SRC_POINTS = [
    [30,  135],   # Top-left  (y ≈ 45% height)
    [270, 135],   # Top-right
    [295, 295],   # Bottom-right
    [5,   295],   # Bottom-left
]

# 1c. Lane Detector — Tham số thuật toán (detect_boundary_path)
# --- EMA Smoothing ---
LANE_EMA_ALPHA          = 0.45   # Hệ số EMA chung [0-1]. 0=giữ cũ, 1=lấy mới hoàn toàn
LANE_EMA_JUMP_THRESHOLD = 40     # Pixel. Nếu C thay đổi hơn mức này → giảm alpha tránh jump
LANE_EMA_JUMP_FACTOR    = 0.3    # Hệ số nhân alpha khi phát hiện jump đột ngột

# --- Boundary Fit Sanity Check ---
LANE_BOUNDARY_A_MAX     = 0.025  # Hệ số cong A tối đa cho biên (quá lớn = nhiễu)
LANE_BOUNDARY_MIN_PTS   = 20     # Số pixel tối thiểu để fit biên
LANE_BOUNDARY_OVERSHOOT = 50     # Pixel. Cho phép fit lố ra ngoài ảnh tối đa bao nhiêu

# --- Contour Filter ---
LANE_CONTOUR_MIN_AREA   = 120    # Pixel². Diện tích contour biên tối thiểu

# --- Dashed Center Line ---
LANE_DASH_AREA_MIN      = 40     # Pixel². Diện tích dash tối thiểu (nhỏ hơn = nhiễu đốm)
LANE_DASH_AREA_MAX      = 3000   # Pixel². Diện tích dash tối đa (lớn hơn = biên, không phải dash)
LANE_DASH_H_MIN         = 5      # Pixel. Chiều cao tối thiểu để coi là dash hợp lệ
LANE_DASH_ASPECT_MAX    = 5.0    # Tỉ lệ w/h tối đa (quá ngang → loại)
LANE_DASH_MIN_COUNT     = 1      # Số đoạn dash tối thiểu để xác nhận nét đứt
LANE_DASH_ALIGN_TOL     = 40     # Pixel. Dung sai căn thẳng hàng X giữa các dash
LANE_DASH_CENTER_LO     = 0.22   # Tỉ lệ width. Cạnh trái vùng scan nét đứt
LANE_DASH_CENTER_HI     = 0.78   # Tỉ lệ width. Cạnh phải vùng scan nét đứt
LANE_DASH_VALID_LO      = 0.15   # Tỉ lệ width. Vùng X hợp lệ tại đáy ảnh (sanity check)
LANE_DASH_VALID_HI      = 0.85   # Tỉ lệ width. Vùng X hợp lệ tại đáy ảnh (sanity check)
LANE_DASH_A_MAX         = 0.03   # Hệ số cong A tối đa cho nét đứt
LANE_DASH_MIN_PTS       = 10     # Số pixel tối thiểu để fit nét đứt
LANE_DASH_LOST_TIMEOUT  = 15     # Frame. Sau bao nhiêu frame mất nét đứt thì reset EMA
LANE_DASH_EMA_JUMP_THR  = 35     # Pixel. Ngưỡng jump C riêng cho nét đứt
LANE_DASH_EMA_JUMP_FAC  = 0.25   # Hệ số nhân alpha khi phát hiện jump nét đứt
LANE_DASH_BOUNDARY_MARGIN = 35   # Pixel. Khoảng cách tối thiểu từ dash đến biên (cross-check)

# --- Image Enhancement ---
LANE_ENHANCE_GAMMA_TARGET  = 128   # Mức sáng trung bình lý tưởng (0-255)
LANE_ENHANCE_GAMMA_MIN     = 0.4   # Gamma tối thiểu (tránh tối quá mức)
LANE_ENHANCE_GAMMA_MAX     = 2.5   # Gamma tối đa (tránh sáng quá mức)
LANE_ENHANCE_CLAHE_CLIP    = 2.5   # CLAHE clipLimit (cao hơn = tương phản mạnh hơn)
LANE_ENHANCE_CLAHE_GRID    = 4     # CLAHE tileGridSize (nhỏ hơn = cục bộ hơn)
LANE_ENHANCE_BILATERAL_D   = 5     # Bilateral filter diameter (lớn hơn = mượt hơn, chậm hơn)
LANE_ENHANCE_BILATERAL_SC  = 60    # Bilateral sigmaColor (lớn = chấp nhận màu khác nhau hơn)
LANE_ENHANCE_BILATERAL_SS  = 60    # Bilateral sigmaSpace (lớn = ảnh hưởng xa hơn)


# 2. Control (PID Controller)
PID_KP = 1.0 # mặc định 0.5
PID_KI = 0.0
PID_KD = 0.1
BASE_SPEED = 0.55        # Tốc độ cơ bản
MAX_THROTTLE = 0.65      # Giới hạn tốc độ
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

# Tự động hiển thị toàn bộ bộ thông số cấu hình mặc định khi bắt đầu chạy chương trình
print("==================== THÔNG SỐ CẤU HÌNH MẶC ĐỊNH ====================", flush=True)
for key, val in sorted(list(globals().items())):
    if key.isupper():
        print(f"  {key:<30} = {val}", flush=True)
print("====================================================================", flush=True)
