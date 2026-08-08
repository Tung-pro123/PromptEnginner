import cv2
import numpy as np
import sys
import os
from src.perception.camera.utils import lane_utils

class LaneDetector:
    """
    Công cụ xử lý ảnh Computer Vision truyền thống để:
    1. Tìm biên trái (Left line) và biên phải (Right line) bằng thuật toán Sliding Window.
    2. Phân đoạn (Segmentation): Tô đa giác màu xanh lá cây vào khu vực "Drivable Area" (Trong lane).
    3. Trả về tọa độ Center để điều khiển xe.
    4. [MỚI] detect_boundary_path: Bám trực tiếp vào đường biên, offset vào trong để tạo quỹ đạo xe.
    """
    def __init__(self, image_width=300, image_height=300):
        self.width = image_width
        self.height = image_height

        # ==========================================
        # KHỞI TẠO MÔ HÌNH ESPCN SUPER RESOLUTION VIA UTILS
        # ==========================================
        self.sr_scale = 2
        self.sr = lane_utils.init_espcn(self.sr_scale)

        # Đọc cấu hình BEV_SRC_POINTS từ settings nếu có
        try:
            _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from src.config import settings as _cfg
            _raw = getattr(_cfg, 'BEV_SRC_POINTS', None)
            _src = np.float32(_raw) if _raw is not None else None
        except Exception:
            _src = None

        # ==========================================
        # EMA COEFFICIENTS SMOOTHING (cho detect_boundary_path)
        # ==========================================
        # Đọc từ settings — load một lần để dùng lại trong toàn bộ class
        try:
            from src.config import settings as _s
        except ImportError:
            _s = None

        def _cfg(key, default):
            return getattr(_s, key, default) if _s is not None else default

        self._ema_alpha           = _cfg('LANE_EMA_ALPHA',          0.45)
        self._ema_jump_threshold  = _cfg('LANE_EMA_JUMP_THRESHOLD', 40)
        self._ema_jump_factor     = _cfg('LANE_EMA_JUMP_FACTOR',    0.3)

        self._boundary_a_max      = _cfg('LANE_BOUNDARY_A_MAX',     0.025)
        self._boundary_min_pts    = _cfg('LANE_BOUNDARY_MIN_PTS',   20)
        self._boundary_overshoot  = _cfg('LANE_BOUNDARY_OVERSHOOT', 50)
        self._contour_min_area    = _cfg('LANE_CONTOUR_MIN_AREA',   120)

        self._dash_area_min       = _cfg('LANE_DASH_AREA_MIN',      40)
        self._dash_area_max       = _cfg('LANE_DASH_AREA_MAX',      3000)
        self._dash_h_min          = _cfg('LANE_DASH_H_MIN',         5)
        self._dash_aspect_max     = _cfg('LANE_DASH_ASPECT_MAX',    5.0)
        self._dash_min_count      = _cfg('LANE_DASH_MIN_COUNT',     2)
        self._dash_align_tol      = _cfg('LANE_DASH_ALIGN_TOL',     40)
        self._dash_center_lo      = _cfg('LANE_DASH_CENTER_LO',     0.22)
        self._dash_center_hi      = _cfg('LANE_DASH_CENTER_HI',     0.78)
        self._dash_valid_lo       = _cfg('LANE_DASH_VALID_LO',      0.15)
        self._dash_valid_hi       = _cfg('LANE_DASH_VALID_HI',      0.85)
        self._dash_a_max          = _cfg('LANE_DASH_A_MAX',         0.03)
        self._dash_min_pts        = _cfg('LANE_DASH_MIN_PTS',       10)
        self._dash_ema_jump_thr   = _cfg('LANE_DASH_EMA_JUMP_THR',  35)
        self._dash_ema_jump_fac   = _cfg('LANE_DASH_EMA_JUMP_FAC',  0.25)

        self._gamma_target        = _cfg('LANE_ENHANCE_GAMMA_TARGET', 128)
        self._gamma_min           = _cfg('LANE_ENHANCE_GAMMA_MIN',    0.4)
        self._gamma_max           = _cfg('LANE_ENHANCE_GAMMA_MAX',    2.5)
        self._clahe_clip          = _cfg('LANE_ENHANCE_CLAHE_CLIP',   2.5)
        self._clahe_grid          = _cfg('LANE_ENHANCE_CLAHE_GRID',   4)
        self._bilateral_d         = _cfg('LANE_ENHANCE_BILATERAL_D',  5)
        self._bilateral_sc        = _cfg('LANE_ENHANCE_BILATERAL_SC', 60)
        self._bilateral_ss        = _cfg('LANE_ENHANCE_BILATERAL_SS', 60)

        # Làm mượt hệ số đa thức A, B, C qua nhiều frame để tránh rung lắc
        self._ema_boundary_fit = None   # fit trực tiếp vào biên đường
        self._last_good_boundary_fit = None  # Hệ số fit gần nhất còn hiệu lực
        self._last_good_boundary_side = 'right'
        self._last_boundary_side = 'right'

        # EMA cho đường trung tâm cũ (để tương thích ngược)
        self._ema_center_fit = None
        self._last_good_center_fit = None
        self._center_detected = False

        # EMA cho nét đứt trung tâm
        self._ema_dash_fit = None
        self._last_good_dash_fit = None
        self._dash_detected = False
        self._dash_lost_frames = 0
        self._dash_boundary_margin = _cfg('LANE_DASH_BOUNDARY_MARGIN', 35)
        self._dash_lost_timeout    = _cfg('LANE_DASH_LOST_TIMEOUT',    15)

        # Bird's Eye View - 4 điểm nguồn (góc phối cảnh) và 4 điểm đích (nhìn từ trên xuống)
        # Nếu BEV_SRC_POINTS được định nghĩa trong settings.py → dùng ngay.
        # Ngược lại dùng giá trị mặc định (calibrate thêm sau trên xe thật).
        if _src is not None and _src.shape == (4, 2):
            self._bev_src = _src
        else:
            self._bev_src = np.float32([
                [30,  int(image_height * 0.45)],               # Top-left
                [image_width - 30, int(image_height * 0.45)],  # Top-right
                [image_width - 5,  image_height - 5],          # Bottom-right
                [5,   image_height - 5],                       # Bottom-left
            ])
        self._bev_dst = np.float32([
            [50,  0],
            [image_width - 50,  0],
            [image_width - 50,  image_height - 1],
            [50,  image_height - 1],
        ])
        self._M    = cv2.getPerspectiveTransform(self._bev_src, self._bev_dst)
        self._Minv = cv2.getPerspectiveTransform(self._bev_dst, self._bev_src)


    # ------------------------------------------------------------------
    # PUBLIC API MỚI: BOUNDARY PATH FOLLOWING
    # ------------------------------------------------------------------

    def detect_boundary_path(self, frame, boundary_offset_px=55, debug=True):
        """
        Thuật toán nâng cấp: Ưu tiên bám nét đứt trung tâm (center dashed line),
        fallback về offset biên (boundary) khi mất nét đứt.

        Priority:
          MODE A — Dashed Center (cao nhất): Phát hiện nét đứt cam/đỏ ở vùng giữa ảnh
                   → Bám thẳng vào nét đứt, đường đi màu TRẮNG trong debug
          MODE B — Boundary Offset (fallback chính): Dùng biên đường + offset vào trong
                   → Đường đi màu XANH LÁ trong debug
          MODE C — Fallback: Tâm ảnh (khi mất cả 2)
                   → Đường đi màu ĐỎ trong debug

        Pipeline:
          1. Image Enhancement (ESPCN + AutoGamma + CLAHE + Bilateral)
          2. BEV warp
          3. HSV Mask lấy vạch đỏ/cam
          4. Contour extraction → lọc contour biên lớn nhất
          4b. [MỚI] Detect nét đứt trung tâm (vị trí + hình dạng)
          5. Polynomial fit + EMA biên
          5b. Polynomial fit + EMA nét đứt
          6. Xác định offset_sign
          7. Tính waypoints theo mode priority
          8. Warp về camera space
          9. Vẽ debug overlay 3 màu

        Returns:
          center_x       : float - tọa độ X tại y=target_y, dùng để tính steering
          waypoints      : list[(x, y)] - điểm dọc đường quỹ đạo (camera space)
          boundary_wps   : list[(x, y)] - điểm dọc đường biên (camera space)
          dash_detected  : bool - True nếu đang ở Mode A (bám nét đứt)
          debug_img      : np.ndarray (BGR) - ảnh gốc đã vẽ overlay
          bev_debug_img  : np.ndarray (BGR) - ảnh Bird's Eye View debug
        """
        if frame is None:
            return float(self.width // 2), [], [], False, None, None

        # -----------------------------------------------------------
        # BƯỚC 1: ESPCN Enhance + Bird's Eye View
        # -----------------------------------------------------------
        resized = lane_utils.enhance_image(frame, self.sr, self.sr_scale, self.width, self.height, self)
        bev     = cv2.warpPerspective(resized, self._M, (self.width, self.height))

        # -----------------------------------------------------------
        # BƯỚC 2: HSV Mask - Lọc vạch đỏ/cam trên BEV
        # -----------------------------------------------------------
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)

        # Dải 1: Đỏ nhạt → cam (H: 0-22)
        lower1 = np.array([0,   85, 40])
        upper1 = np.array([22, 255, 255])
        mask1  = cv2.inRange(hsv, lower1, upper1)

        # Dải 2: Đỏ đậm (H: 160-180)
        lower2 = np.array([160, 85, 40])
        upper2 = np.array([180, 255, 255])
        mask2  = cv2.inRange(hsv, lower2, upper2)

        red_mask = cv2.bitwise_or(mask1, mask2)

        # ROI: Bỏ qua 30% phía trên (chứa nền xa, không liên quan)
        roi_mask = np.zeros_like(red_mask)
        roi_mask[int(self.height * 0.30):, :] = 255
        red_mask = cv2.bitwise_and(red_mask, roi_mask)

        # Morphology: Lọc nhiễu nhỏ (Open) + Khép lỗ đứt gãy (Close)
        k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  k_open)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k_close)

        # -----------------------------------------------------------
        # BƯỚC 3: Contour extraction + Lọc nhiễu thông minh
        # -----------------------------------------------------------
        cnts_data = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours  = cnts_data[0] if len(cnts_data) == 2 else cnts_data[1]

        # 3a. Lọc theo diện tích tối thiểu
        MIN_AREA = self._contour_min_area
        valid_cnts = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

        # 3b. Lọc theo tỷ lệ khung hình (aspect ratio): Đường biên đường phải dọc
        #     Loại bỏ contour quá ngang (nền trần nhà, đèn đường...)
        def _is_road_boundary_contour(cnt):
            x, y, w, h = cv2.boundingRect(cnt)
            if h < 10:           # Quá mỏng theo chiều dọc
                return False
            aspect = w / float(h) if h > 0 else 99
            if aspect > 4.0:     # Quá ngang (ngang/dọc > 4:1) → loại
                return False
            # Vùng y phải nằm trong ROI thực tế (bỏ top 30%)
            if y + h < int(self.height * 0.30):
                return False
            return True

        valid_cnts = [c for c in valid_cnts if _is_road_boundary_contour(c)]

        # 3c. Tách contour sang trái và phải theo vị trí centroid
        cx_mid = self.width / 2.0
        left_cnts  = []
        right_cnts = []
        for c in valid_cnts:
            M = cv2.moments(c)
            if M['m00'] > 0:
                cx_c = M['m10'] / M['m00']
                if cx_c < cx_mid:
                    left_cnts.append((cv2.contourArea(c), c))
                else:
                    right_cnts.append((cv2.contourArea(c), c))

        # 3d. Chỉ giữ lại contour lớn nhất mỗi bên (tránh nhiễu đốm nhỏ)
        best_left  = max(left_cnts,  key=lambda x: x[0])[1] if left_cnts  else None
        best_right = max(right_cnts, key=lambda x: x[0])[1] if right_cnts else None

        # 3e. Quyết định chọn biên nào để fit
        #     - Ưu tiên bên phải (thường là vạch đỏ biên đường trong track)
        #     - Chỉ dùng biên trái nếu không có bên phải
        #     - Nếu có cả 2: ưu tiên cái có diện tích lớn hơn (biên thật)
        boundary_pts_x = np.array([], dtype=np.float32)
        boundary_pts_y = np.array([], dtype=np.float32)
        boundary_side  = 'none'

        area_left  = left_cnts[0][0]  if left_cnts  else 0
        area_right = right_cnts[0][0] if right_cnts else 0

        chosen_cnt = None
        if best_right is not None and best_left is not None:
            # Cả 2 bên → chọn bên có diện tích lớn hơn đáng kể (>1.5x)
            if area_right >= area_left * 1.5:
                chosen_cnt    = best_right
                boundary_side = 'right'
            elif area_left >= area_right * 1.5:
                chosen_cnt    = best_left
                boundary_side = 'left'
            else:
                # Gần nhau → dùng bên phải (vạch biên chuẩn)
                chosen_cnt    = best_right
                boundary_side = 'right'
        elif best_right is not None:
            chosen_cnt    = best_right
            boundary_side = 'right'
        elif best_left is not None:
            chosen_cnt    = best_left
            boundary_side = 'left'

        # Ghi nhớ bên biên gần nhất để ưu tiên trong các frame sau
        if boundary_side in ('left', 'right'):
            self._last_boundary_side = boundary_side

        if chosen_cnt is not None:
            pts = chosen_cnt.reshape(-1, 2)
            boundary_pts_x = pts[:, 0].astype(np.float32)
            boundary_pts_y = pts[:, 1].astype(np.float32)

        # -----------------------------------------------------------
        # BƯỚC 4: Polynomial fit + Sanity check chống nhiễu cực đoan
        # -----------------------------------------------------------
        boundary_fit = None
        MIN_PTS_FOR_FIT = self._boundary_min_pts
        A_MAX = self._boundary_a_max

        if len(boundary_pts_y) >= MIN_PTS_FOR_FIT:
            try:
                candidate_fit = np.polyfit(boundary_pts_y, boundary_pts_x, 2)

                # --- Kiểm tra sanity: Hệ số A không được quá lớn ---
                A_coeff = abs(candidate_fit[0])
                if A_coeff > A_MAX:
                    # Đường cong quá cực đoan → là nhiễu, từ chối
                    candidate_fit = None

                # --- Kiểm tra: Giá trị C (vị trí X tại y=0) phải hợp lý ---
                if candidate_fit is not None:
                    x_at_bottom = (candidate_fit[0] * (self.height - 1)**2
                                   + candidate_fit[1] * (self.height - 1)
                                   + candidate_fit[2])
                    if x_at_bottom < -self._boundary_overshoot or x_at_bottom > self.width + self._boundary_overshoot:
                        # Đường fit ra ngoài ảnh quá nhiều → nhiễu
                        candidate_fit = None

                if candidate_fit is not None:
                    # EMA smoothing hệ số fit
                    if self._ema_boundary_fit is None:
                        self._ema_boundary_fit = candidate_fit.copy()
                    else:
                        # Kiểm tra jump đột ngột: Nếu C thay đổi quá lớn so với EMA → giảm alpha
                        delta_c = abs(candidate_fit[2] - self._ema_boundary_fit[2])
                        alpha   = self._ema_alpha if delta_c < self._ema_jump_threshold else (self._ema_alpha * self._ema_jump_factor)
                        self._ema_boundary_fit = (
                            alpha * candidate_fit
                            + (1.0 - alpha) * self._ema_boundary_fit
                        )
                    boundary_fit = self._ema_boundary_fit.copy()
                    self._last_good_boundary_fit = boundary_fit.copy()
                    self._last_good_boundary_side = boundary_side # Ghi nhớ side tương ứng với fit hợp lệ

            except (np.linalg.LinAlgError, ValueError):
                boundary_fit = None

        # Dùng fit gần nhất nếu frame này không có (hoặc bị từ chối)
        if boundary_fit is None and self._last_good_boundary_fit is not None:
            boundary_fit = self._last_good_boundary_fit.copy()
            boundary_side = self._last_good_boundary_side # Phục hồi side tương ứng với fit cũ


        # -----------------------------------------------------------
        # BƯỚC 4b: [MỚI] Phát hiện nét đứt trung tâm (Dashed Center Line)
        # -----------------------------------------------------------
        # Dùng raw red_mask (trước khi morphology Close lớn) để giữ tính gián đoạn của dash.
        # Tạo raw_mask_open: chỉ dùng morphology Open nhỏ để lọc nhiễu đốm, KHÔNG Close.
        k_open_dash = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        red_mask_for_dash = cv2.morphologyEx(
            cv2.bitwise_and(cv2.bitwise_or(mask1, mask2), roi_mask),
            cv2.MORPH_OPEN, k_open_dash
        )
        dash_pts_x, dash_pts_y, dash_raw_detected = lane_utils.detect_dashed_center(
            hsv, red_mask_for_dash, self.width, self.height, self, boundary_fit, boundary_side
        )

        # Polyfit nét đứt + EMA smoothing
        dash_fit = None
        self._dash_detected = False

        if dash_raw_detected and len(dash_pts_y) >= self._dash_min_pts:
            try:
                candidate_dash = np.polyfit(dash_pts_y, dash_pts_x, 2)

                # Sanity check hệ số A
                if abs(candidate_dash[0]) <= self._dash_a_max:
                    # Sanity check vị trí X tại đáy ảnh (phải ở vùng trung tâm)
                    x_dash_bottom = (candidate_dash[0] * (self.height - 1)**2
                                     + candidate_dash[1] * (self.height - 1)
                                     + candidate_dash[2])
                    if int(self.width * self._dash_valid_lo) < x_dash_bottom < int(self.width * self._dash_valid_hi):
                        # [MỚI] Cross-check: Dash phải cách biên ít nhất _dash_boundary_margin pixel
                        # Tránh trường hợp biên cong vào vùng trung tâm bị nhầm thành nét đứt
                        dash_too_close_to_boundary = False
                        if boundary_fit is not None:
                            ref_y = int(self.height * 0.7)  # Kiểm tra tại y=70% chiều cao
                            x_boundary_ref = (boundary_fit[0]*ref_y**2
                                              + boundary_fit[1]*ref_y
                                              + boundary_fit[2])
                            if abs(x_dash_bottom - x_boundary_ref) < self._dash_boundary_margin:
                                dash_too_close_to_boundary = True

                        if not dash_too_close_to_boundary:
                            # EMA smoothing
                            if self._ema_dash_fit is None:
                                self._ema_dash_fit = candidate_dash.copy()
                            else:
                                delta_c = abs(candidate_dash[2] - self._ema_dash_fit[2])
                                alpha = self._ema_alpha if delta_c < self._dash_ema_jump_thr else (self._ema_alpha * self._dash_ema_jump_fac)
                                self._ema_dash_fit = (alpha * candidate_dash
                                                      + (1.0 - alpha) * self._ema_dash_fit)
                            dash_fit = self._ema_dash_fit.copy()
                            self._last_good_dash_fit = dash_fit.copy()
                            self._dash_detected = True
                            self._dash_lost_frames = 0

            except (np.linalg.LinAlgError, ValueError):
                pass

        # Quản lý timeout nét đứt: Nếu mất quá lâu → reset EMA để không bám fit cũ sai
        if not self._dash_detected:
            self._dash_lost_frames += 1
            if self._dash_lost_frames >= self._dash_lost_timeout:
                self._ema_dash_fit = None       # Reset EMA
                self._last_good_dash_fit = None # Xóa cả last good để không dùng fallback sai

        # -----------------------------------------------------------
        # BƯỚC 5: Offset đường fit vào trong làn đường
        # -----------------------------------------------------------
        # Xác định hướng offset: Biên bên phải → offset sang trái (-)
        #                         Biên bên trái  → offset sang phải (+)
        if boundary_side == 'right':
            offset_sign = -1   # offset vào trong = sang trái
        elif boundary_side == 'left':
            offset_sign = +1   # offset vào trong = sang phải
        else:
            # Không biết bên nào → thử dùng đường trung tâm ảnh làm tham chiếu
            offset_sign = -1 if (len(boundary_pts_x) > 0 and np.mean(boundary_pts_x) > self.width / 2) else +1

        # -----------------------------------------------------------
        # BƯỚC 6: Dark Road Center Assist - Tìm vùng đường tối làm cross-check
        # -----------------------------------------------------------
        # BƯỚC 7: Tính 60 waypoints — Priority Logic 3 Mode
        # -----------------------------------------------------------
        # MODE A: Bám nét đứt trung tâm (ưu tiên cao nhất)
        # MODE B: Offset từ biên đường (fallback khi mất nét đứt)
        # MODE C: Giữ fit cũ / tâm ảnh (fallback cuối cùng)
        # -----------------------------------------------------------
        y_lines_bev = np.linspace(int(self.height * 0.40), self.height - 15, 60).astype(int)
        waypoints_bev = []
        boundary_waypoints_bev = []

        # Xác định mode hoạt động
        if self._dash_detected and dash_fit is not None:
            active_mode = 'A_DASH'       # Bám trực tiếp nét đứt
        elif boundary_fit is not None:
            active_mode = 'B_BOUNDARY'   # Offset từ biên
        else:
            active_mode = 'C_FALLBACK'   # Fallback

        for y in y_lines_bev:
            # --- Tính vị trí biên để vẽ debug ---
            if boundary_fit is not None:
                x_boundary = float(boundary_fit[0]*y**2 + boundary_fit[1]*y + boundary_fit[2])
                x_boundary = float(np.clip(x_boundary, 5, self.width - 5))
            else:
                x_boundary = float(self.width // 2)
            boundary_waypoints_bev.append((int(x_boundary), int(y)))

            # --- MODE A: Bám thẳng nét đứt trung tâm ---
            if active_mode == 'A_DASH':
                x_path = float(dash_fit[0]*y**2 + dash_fit[1]*y + dash_fit[2])
                x_path = float(np.clip(x_path, 5, self.width - 5))

            # --- MODE B: Offset vào trong từ biên ---
            elif active_mode == 'B_BOUNDARY':
                x_path = x_boundary + offset_sign * boundary_offset_px
                x_path = float(np.clip(x_path, 5, self.width - 5))

            # --- MODE C: Fallback tâm ảnh ---
            else:
                x_path = float(self.width // 2)

            waypoints_bev.append((int(x_path), int(y)))

        # -----------------------------------------------------------
        # BƯỚC 8: Warp waypoints từ BEV về camera space
        # -----------------------------------------------------------
        waypoints_cam = self._warp_points_back(waypoints_bev)
        boundary_waypoints_cam = self._warp_points_back(boundary_waypoints_bev)

        # Lấy center_x tại y gần mũi xe nhất (y_bev lớn nhất → y_cam sát dưới)
        if waypoints_cam:
            center_x = float(waypoints_cam[-1][0])
        else:
            center_x = float(self.width // 2)

        # -----------------------------------------------------------
        # BƯỚC 9: Vẽ debug overlay — 3 màu phân biệt mode
        # -----------------------------------------------------------
        # Màu path theo mode:
        #   Mode A (nét đứt)  → Trắng  (255, 255, 255)
        #   Mode B (biên)      → Xanh lá (0, 220, 60)
        #   Mode C (fallback)  → Đỏ     (0, 0, 200)
        MODE_COLORS = {
            'A_DASH':     (255, 255, 255),   # Trắng — bám nét đứt
            'B_BOUNDARY': (0, 220, 60),      # Xanh lá — offset biên
            'C_FALLBACK': (0, 0, 200),       # Đỏ — fallback
        }
        path_color = MODE_COLORS[active_mode]

        debug_img     = None
        bev_debug_img = None

        if debug:
            debug_img     = resized.copy()
            bev_debug_img = bev.copy()

            # --- Vẽ mask biên đỏ lên BEV ---
            red_overlay = bev_debug_img.copy()
            red_overlay[red_mask > 0] = [0, 80, 255]
            bev_debug_img = cv2.addWeighted(bev_debug_img, 0.6, red_overlay, 0.4, 0)

            # --- Vẽ đường fit biên (màu cam sáng) ---
            if boundary_fit is not None:
                ploty = np.linspace(int(self.height * 0.30), self.height - 1, 60).astype(int)
                for y in ploty:
                    x_b = int(boundary_fit[0]*y**2 + boundary_fit[1]*y + boundary_fit[2])
                    x_b = int(np.clip(x_b, 0, self.width - 1))
                    cv2.circle(bev_debug_img, (x_b, y), 2, (0, 165, 255), -1)

            # --- [MỚI] Vẽ đường fit nét đứt (màu tím) ---
            if dash_fit is not None:
                ploty = np.linspace(int(self.height * 0.30), self.height - 1, 60).astype(int)
                for y in ploty:
                    x_d = int(dash_fit[0]*y**2 + dash_fit[1]*y + dash_fit[2])
                    x_d = int(np.clip(x_d, 0, self.width - 1))
                    cv2.circle(bev_debug_img, (x_d, y), 2, (200, 0, 200), -1)

            # --- [MỚI] Vẽ các điểm nét đứt thô (màu vàng nhạt) ---
            if dash_raw_detected and len(dash_pts_x) > 0:
                for px, py in zip(dash_pts_x.astype(int), dash_pts_y.astype(int)):
                    if 0 <= px < self.width and 0 <= py < self.height:
                        cv2.circle(bev_debug_img, (px, py), 1, (0, 230, 230), -1)

            # --- Vẽ waypoints path trên BEV ---
            for pt in waypoints_bev:
                cv2.circle(bev_debug_img, pt, 4, path_color, -1)
            if len(waypoints_bev) > 1:
                for i in range(1, len(waypoints_bev)):
                    cv2.line(bev_debug_img, waypoints_bev[i-1], waypoints_bev[i], path_color, 2)

            # --- Vẽ trục tâm BEV (xanh dương) ---
            cv2.line(bev_debug_img, (self.width // 2, 0), (self.width // 2, self.height), (255, 100, 0), 1)

            # --- Label mode và biên ---
            mode_label = {
                'A_DASH':     'MODE A: DASH CENTER',
                'B_BOUNDARY': f'MODE B: BOUNDARY {boundary_side.upper()}',
                'C_FALLBACK': 'MODE C: FALLBACK',
            }[active_mode]
            cv2.putText(bev_debug_img, mode_label, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, path_color, 1)
            cv2.putText(bev_debug_img, "BEV (Bird Eye View)", (5, self.height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

            # --- Vẽ waypoints đã warp về camera space ---
            for pt in waypoints_cam:
                cv2.circle(debug_img, pt, 4, path_color, -1)
            if len(waypoints_cam) > 1:
                for i in range(1, len(waypoints_cam)):
                    cv2.line(debug_img, waypoints_cam[i-1], waypoints_cam[i], path_color, 2)

            # --- Mũi tên center_x ---
            arrow_y = self.height - 30
            cv2.arrowedLine(debug_img,
                            (self.width // 2, arrow_y),
                            (int(center_x), arrow_y),
                            (0, 200, 255), 2, tipLength=0.35)

            # --- Label mode trên camera frame ---
            cv2.putText(debug_img, mode_label, (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, path_color, 1)

        return center_x, waypoints_cam, boundary_waypoints_cam, self._dash_detected, debug_img, bev_debug_img


    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------



    def _warp_points_back(self, pts_bev):
        """
        Chuyển danh sách điểm từ Bird's Eye View space về camera space gốc.
        pts_bev: list[(x, y)] trong BEV
        Returns: list[(x, y)] trong camera frame
        """
        if not pts_bev:
            return []
        pts_arr = np.float32(pts_bev).reshape(-1, 1, 2)
        pts_cam = cv2.perspectiveTransform(pts_arr, self._Minv)
        result  = []
        for p in pts_cam.reshape(-1, 2):
            x = int(np.clip(p[0], 0, self.width  - 1))
            y = int(np.clip(p[1], 0, self.height - 1))
            result.append((x, y))
        return result

    # ------------------------------------------------------------------
    # LEGACY METHODS (Giữ nguyên để không phá vỡ code cũ)
    # ------------------------------------------------------------------

    def process_and_segment(self, frame, threshold_val=200):
        """
        Nhận vào ảnh Gốc, trả về:
        - segmented_img: Ảnh gốc đã được "tô màu" vùng đi được (Drivable area overlay).
        - thresh_img: Ảnh nhị phân sau khi lọc để debug.
        - center_x: Tâm điểm làn đường để gửi tới controller.
        - left_fit, right_fit: Tham số đa thức bậc 2 của hai lề đường.
        """
        if frame is None:
            return None, None, self.width // 2, None, None
            
        # ==========================================
        # 1. TIỀN XỬ LÝ (PRE-PROCESSING) VỚI ESPCN
        # ==========================================
        resized = lane_utils.enhance_image(frame, self.sr, self.sr_scale, self.width, self.height, self)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Làm mờ và tăng tương phản (CLAHE) - Kế thừa kỹ thuật cũ để lọc nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Nhị phân hóa (Lọc lấy những vệt sáng trắng như vạch đường)
        _, thresh = cv2.threshold(enhanced, threshold_val, 255, cv2.THRESH_BINARY)
        
        # ==========================================
        # 2. SLIDING WINDOW & SEGMENTATION VIA UTILS
        # ==========================================
        segmented_img, center_x, left_fit, right_fit = lane_utils.sliding_window_segment(
            thresh, resized, self.width, self.height, limit_y_ratio=0.0
        )
        return segmented_img, thresh, center_x, left_fit, right_fit

    def process_color_segment(self, frame):
        """
        Nhận vào ảnh Gốc, dùng HSV lọc màu Cam/Đỏ để tạo ảnh nhị phân (thresh),
        sau đó áp dụng thuật toán Sliding Window và Curve Fitting (hồi quy đa thức)
        để xác định làn đường và tô đa giác vùng chạy được.
        """
        if frame is None:
            return None, None, self.width // 2, None, None
            
        # ==========================================
        # 1. TIỀN XỬ LÝ ESPCN & HSV (COLOR FILTERING)
        # ==========================================
        resized = self._enhance_image(frame)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Ngưỡng màu Đỏ/Cam (Dải 1: Đỏ nhạt đến cam)
        # Giảm nhẹ Saturation xuống 85 và Value xuống 40 để nhạy hơn với vạch đỏ trong bóng râm, nhưng vẫn đủ cao để lọc nền trắng
        lower_red1 = np.array([0, 85, 40])
        upper_red1 = np.array([22, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        # Ngưỡng màu Đỏ/Cam (Dải 2: Đỏ đậm)
        lower_red2 = np.array([160, 85, 40])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Hợp nhất dải màu để lấy vạch biên
        thresh = cv2.bitwise_or(mask1, mask2)
        
        # 1.1 Áp dụng ROI Mask: Chỉ quét phần đường dưới (bỏ qua 35% phía trên chứa cảnh nền/trần nhà nhiễu)
        roi_mask = np.zeros_like(thresh)
        roi_mask[int(self.height * 0.35):, :] = 255
        thresh = cv2.bitwise_and(thresh, roi_mask)
        
        # 1.2 Lọc nhiễu đốm trắng li ti bằng Morphological Opening
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        
        # 1.3 Làm liền nét/đóng các lỗ đứt gãy nhỏ trên vạch kẻ bằng Morphological Closing
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        
        # 1.4 Dùng Contours lọc nhiễu đốm nhỏ có diện tích dưới 35px
        contours_data = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
        valid_contours = [c for c in contours if cv2.contourArea(c) > 35]
        
        clean_thresh = np.zeros_like(thresh)
        cv2.drawContours(clean_thresh, valid_contours, -1, 255, -1)
        thresh = clean_thresh

        # ==========================================
        # 2. SLIDING WINDOW & SEGMENTATION VIA UTILS
        # ==========================================
        segmented_img, center_x, left_fit, right_fit = lane_utils.sliding_window_segment(
            thresh, resized, self.width, self.height, limit_y_ratio=0.5
        )
        return segmented_img, thresh, center_x, left_fit, right_fit
