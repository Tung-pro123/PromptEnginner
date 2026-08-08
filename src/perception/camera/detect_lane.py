import cv2
import numpy as np
import sys
import os

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
        # Làm mượt hệ số đa thức A, B, C qua nhiều frame để tránh rung lắc
        self._ema_boundary_fit = None   # fit trực tiếp vào biên đường
        self._ema_alpha = 0.45          # 0 = hoàn toàn cũ, 1 = hoàn toàn mới
        self._last_good_boundary_fit = None  # Hệ số fit gần nhất còn hiệu lực
        self._last_boundary_side = 'right'   # Bên biên gần nhất phát hiện được (mặc định phải)

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
        Thuật toán mới: Bám trực tiếp vào đường biên (vạch đỏ/cam) rồi offset vào trong
        để tạo ra quỹ đạo xe song song và cách biên `boundary_offset_px` pixel.

        Pipeline:
          1. Resize + Bird's Eye View warp
          2. HSV Mask lấy vạch đỏ/cam
          3. Contour extraction → lọc contour biên lớn nhất
          4. Polynomial fit bậc 2 trực tiếp vào contour pixels
          5. Offset đường fit vào trong
          6. Dark Road Center assist để cross-check
          7. EMA làm mượt hệ số qua frame
          8. Vẽ debug overlay và trả về kết quả

        Returns:
          center_x      : float - tọa độ X tại y=target_y, dùng để tính steering
          waypoints     : list[(x, y)] - 8 điểm dọc đường quỹ đạo (không phải Bird-eye, đã warp về camera)
          debug_img     : np.ndarray (BGR) - ảnh gốc đã vẽ overlay để debug
          bev_debug_img : np.ndarray (BGR) - ảnh Bird's Eye View debug
        """
        if frame is None:
            return float(self.width // 2), [], None, None

        # -----------------------------------------------------------
        # BƯỚC 1: Resize + Bird's Eye View
        # -----------------------------------------------------------
        resized = cv2.resize(frame, (self.width, self.height))
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
        MIN_AREA = 120
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
        MIN_PTS_FOR_FIT = 20
        # Hệ số cong A tối đa cho phép: |A| > A_MAX là đường bị nhiễu phi thực tế
        A_MAX = 0.025   # Tương đương ~1 pixel lệch / (10px)^2 — đường cong vừa phải

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
                    if x_at_bottom < -50 or x_at_bottom > self.width + 50:
                        # Đường fit ra ngoài ảnh quá nhiều → nhiễu
                        candidate_fit = None

                if candidate_fit is not None:
                    # EMA smoothing hệ số fit
                    if self._ema_boundary_fit is None:
                        self._ema_boundary_fit = candidate_fit.copy()
                    else:
                        # Kiểm tra jump đột ngột: Nếu C thay đổi quá lớn so với EMA → giảm alpha
                        delta_c = abs(candidate_fit[2] - self._ema_boundary_fit[2])
                        alpha   = self._ema_alpha if delta_c < 40 else (self._ema_alpha * 0.3)
                        self._ema_boundary_fit = (
                            alpha * candidate_fit
                            + (1.0 - alpha) * self._ema_boundary_fit
                        )
                    boundary_fit = self._ema_boundary_fit.copy()
                    self._last_good_boundary_fit = boundary_fit.copy()

            except (np.linalg.LinAlgError, ValueError):
                boundary_fit = None

        # Dùng fit gần nhất nếu frame này không có (hoặc bị từ chối)
        if boundary_fit is None and self._last_good_boundary_fit is not None:
            boundary_fit = self._last_good_boundary_fit.copy()


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
        dark_centers = self._compute_dark_road_centers(bev)  # {y: cx}

        # -----------------------------------------------------------
        # BƯỚC 7: Tính 8 waypoints dọc đường quỹ đạo (trong không gian BEV)
        # -----------------------------------------------------------
        y_lines_bev = np.linspace(int(self.height * 0.40), self.height - 15, 8).astype(int)
        waypoints_bev = []  # [(x_bev, y_bev)] - trong không gian Bird's Eye View

        for y in y_lines_bev:
            if boundary_fit is not None:
                x_boundary = float(boundary_fit[0]*y**2 + boundary_fit[1]*y + boundary_fit[2])
                x_path     = x_boundary + offset_sign * boundary_offset_px

                # Cross-check với dark road center: Kéo nhẹ về phía trung tâm đường tối nếu tồn tại
                if y in dark_centers:
                    dc = dark_centers[y]
                    # Blend 20% về phía dark center để tránh lệch ra ngoài làn
                    x_path = 0.80 * x_path + 0.20 * dc

                # Clamp trong biên ảnh có padding nhỏ
                x_path = float(np.clip(x_path, 5, self.width - 5))
                waypoints_bev.append((int(x_path), int(y)))
            else:
                # Fallback: Dùng tâm ảnh
                waypoints_bev.append((self.width // 2, int(y)))

        # -----------------------------------------------------------
        # BƯỚC 8: Warp waypoints từ BEV về camera space
        # -----------------------------------------------------------
        waypoints_cam = self._warp_points_back(waypoints_bev)

        # Lấy center_x tại y gần mũi xe nhất (y_bev lớn nhất → y_cam sát dưới)
        if waypoints_cam:
            center_x = float(waypoints_cam[-1][0])
        else:
            center_x = float(self.width // 2)

        # -----------------------------------------------------------
        # BƯỚC 9: Vẽ debug overlay
        # -----------------------------------------------------------
        debug_img     = None
        bev_debug_img = None

        if debug:
            debug_img     = resized.copy()
            bev_debug_img = bev.copy()

            # Vẽ mask biên đỏ lên BEV debug
            red_overlay = bev_debug_img.copy()
            red_overlay[red_mask > 0] = [0, 80, 255]
            bev_debug_img = cv2.addWeighted(bev_debug_img, 0.6, red_overlay, 0.4, 0)

            # Vẽ đường fit biên (màu cam sáng) trên BEV
            if boundary_fit is not None:
                ploty = np.linspace(int(self.height * 0.30), self.height - 1, 60).astype(int)
                for y in ploty:
                    x_b = int(boundary_fit[0]*y**2 + boundary_fit[1]*y + boundary_fit[2])
                    x_b = int(np.clip(x_b, 0, self.width - 1))
                    cv2.circle(bev_debug_img, (x_b, y), 2, (0, 165, 255), -1)

            # Vẽ 8 waypoints path (màu xanh lá) trên BEV
            for i, pt in enumerate(waypoints_bev):
                cv2.circle(bev_debug_img, pt, 5, (0, 255, 0), -1)
            if len(waypoints_bev) > 1:
                for i in range(1, len(waypoints_bev)):
                    cv2.line(bev_debug_img, waypoints_bev[i-1], waypoints_bev[i], (0, 255, 0), 2)

            # Vẽ trục tâm BEV (xanh dương)
            cv2.line(bev_debug_img, (self.width // 2, 0), (self.width // 2, self.height), (255, 100, 0), 1)

            # Label BEV side
            label_side = f"Boundary: {boundary_side.upper()}"
            cv2.putText(bev_debug_img, label_side, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(bev_debug_img, "BEV (Bird Eye View)", (5, self.height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

            # Vẽ waypoints đã warp về camera space trên debug_img gốc
            for i, pt in enumerate(waypoints_cam):
                cv2.circle(debug_img, pt, 5, (0, 255, 0), -1)
            if len(waypoints_cam) > 1:
                for i in range(1, len(waypoints_cam)):
                    cv2.line(debug_img, waypoints_cam[i-1], waypoints_cam[i], (0, 255, 0), 2)

            # Mũi tên center_x
            arrow_y = self.height - 30
            cv2.arrowedLine(debug_img,
                            (self.width // 2, arrow_y),
                            (int(center_x), arrow_y),
                            (0, 200, 255), 2, tipLength=0.35)

        return center_x, waypoints_cam, debug_img, bev_debug_img

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _compute_dark_road_centers(self, bev_frame):
        """
        Tìm tâm X của vùng đường tối (mặt nhựa/bê tông) tại mỗi scanline y trên BEV.
        Trả về dict {y: center_x}.
        """
        gray   = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2GRAY)
        result = {}
        y_scan = np.linspace(int(self.height * 0.40), self.height - 15, 12).astype(int)

        for y in y_scan:
            if y >= self.height:
                continue
            row = gray[y, :]
            # Threshold: Vùng đường tối (V < 100)
            dark_cols = np.where(row < 100)[0]
            if len(dark_cols) >= 10:
                result[y] = float(np.mean(dark_cols))
        return result

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
        # 1. TIỀN XỬ LÝ (PRE-PROCESSING)
        # ==========================================
        resized = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Làm mờ và tăng tương phản (CLAHE) - Kế thừa kỹ thuật cũ để lọc nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Nhị phân hóa (Lọc lấy những vệt sáng trắng như vạch đường)
        _, thresh = cv2.threshold(enhanced, threshold_val, 255, cv2.THRESH_BINARY)
        
        # ==========================================
        # 2. TÌM CHÂN VẠCH ĐƯỜNG (HISTOGRAM PEAKS)
        # ==========================================
        # Lấy histogram (tổng điểm ảnh trắng theo chiều dọc) của nửa dưới bức ảnh
        # Nửa dưới bức ảnh chứa phần đường gần xe nhất.
        histogram = np.sum(thresh[self.height//2:, :], axis=0)
        
        midpoint = int(histogram.shape[0] // 2)
        # Đỉnh histogram bên trái là chân vạch trái, bên phải là chân vạch phải
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Fallback (Nếu ảnh mù mịt không có vạch, giả định vị trí)
        if histogram[leftx_base] < 50:
            leftx_base = 30
        if histogram[rightx_base] < 50:
            rightx_base = self.width - 30
            
        # ==========================================
        # 3. THUẬT TOÁN SLIDING WINDOW
        # ==========================================
        nwindows = 9  # Chia ảnh thành 9 lớp ngang
        window_height = int(self.height // nwindows)
        
        # Trích xuất tất cả tọa độ pixel trắng (X, Y)
        nonzero = thresh.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 35  # Độ rộng cửa sổ dò tìm
        minpix = 15  # Số pixel trắng tối thiểu để dịch chuyển tâm cửa sổ
        
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(nwindows):
            # Tính giới hạn Y của cửa sổ hiện tại (Dò từ dưới lên trên)
            win_y_low = self.height - (window + 1) * window_height
            win_y_high = self.height - window * window_height
            
            # Tính giới hạn X cho cửa sổ trái và phải
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
            
            # Lọc lấy các pixel trắng rơi vào bên trong cửa sổ
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # Cập nhật lại tâm của cửa sổ nếu gom đủ lượng pixel trắng
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        
        # ==========================================
        # 4. HỒI QUY ĐA THỨC (CURVE FITTING) VÀ PHÂN ĐOẠN (SEGMENTATION)
        # ==========================================
        left_fit, right_fit = None, None
        segmented_img = resized.copy()
        
        # Chỉ chạy nội suy khi có đủ số điểm ảnh làm cơ sở
        if len(leftx) > 10:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 10:
            right_fit = np.polyfit(righty, rightx, 2)
            
        # Ước lượng độ rộng làn đường (mặc định khoảng 140 pixel cho ảnh 300x300)
        lane_width = 140.0
        
        # Nếu chỉ tìm thấy một bên, giả lập bên còn lại song song bằng cách dịch chuyển theo lane_width
        if left_fit is not None and right_fit is None:
            right_fit = left_fit.copy()
            right_fit[2] += lane_width
        elif right_fit is not None and left_fit is None:
            left_fit = right_fit.copy()
            left_fit[2] -= lane_width
            
        center_x = self.width // 2
        
        if left_fit is not None and right_fit is not None:
            # Nội suy tọa độ X từ 0 đến height
            ploty = np.linspace(0, self.height - 1, self.height)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            # --- ĐỔ MÀU PHÂN ĐOẠN LÀN ĐƯỜNG (DRIVABLE AREA) ---
            # Tập hợp tọa độ bên trái và bên phải để tạo thành 1 Đa giác khép kín
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) # Lật mảng phải để khép hình
            pts = np.hstack((pts_left, pts_right))
            
            # Phân đoạn: Phủ một lớp màu Xanh Lá Cây 30% alpha (Blend) vào ảnh gốc
            color_mask = np.zeros_like(segmented_img)
            cv2.fillPoly(color_mask, np.int_([pts]), (0, 255, 0)) # Tô màu xanh lá (0, 255, 0)
            segmented_img = cv2.addWeighted(segmented_img, 1.0, color_mask, 0.4, 0) # 0.4 là độ đậm nhạt
            
            # Tính điểm mục tiêu để lái xe (Chính giữa vạch trái và vạch phải ở sát đầu xe)
            target_y = self.height - 20 # Sát mũi xe
            lx = left_fit[0]*target_y**2 + left_fit[1]*target_y + left_fit[2]
            rx = right_fit[0]*target_y**2 + right_fit[1]*target_y + right_fit[2]
            center_x = (lx + rx) / 2.0
            
            # Vẽ điểm lái màu đỏ
            cv2.circle(segmented_img, (int(center_x), target_y), 6, (0, 0, 255), -1)

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
        # 1. TIỀN XỬ LÝ BẰNG HSV (COLOR FILTERING)
        # ==========================================
        resized = cv2.resize(frame, (self.width, self.height))
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
        # 2. TÌM CHÂN VẠCH ĐƯỜNG (HISTOGRAM PEAKS)
        # ==========================================
        histogram = np.sum(thresh[self.height//2:, :], axis=0)
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        if histogram[leftx_base] < 50:
            leftx_base = 30
        if histogram[rightx_base] < 50:
            rightx_base = self.width - 30
            
        # ==========================================
        # 3. THUẬT TOÁN SLIDING WINDOW
        # ==========================================
        nwindows = 9
        window_height = int(self.height // nwindows)
        
        nonzero = thresh.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 35
        minpix = 15
        
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(nwindows):
            win_y_low = self.height - (window + 1) * window_height
            win_y_high = self.height - window * window_height
            
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        
        # ==========================================
        # 4. HỒI QUY ĐA THỨC VÀ PHÂN ĐOẠN
        # ==========================================
        left_fit, right_fit = None, None
        segmented_img = resized.copy()
        
        if len(leftx) > 10:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 10:
            right_fit = np.polyfit(righty, rightx, 2)
            
        # Ước lượng độ rộng làn đường (mặc định khoảng 140 pixel cho ảnh 300x300)
        lane_width = 140.0
        
        # Nếu chỉ tìm thấy một bên, giả lập bên còn lại song song bằng cách dịch chuyển theo lane_width
        if left_fit is not None and right_fit is None:
            right_fit = left_fit.copy()
            right_fit[2] += lane_width
        elif right_fit is not None and left_fit is None:
            left_fit = right_fit.copy()
            left_fit[2] -= lane_width
            
        center_x = self.width // 2
        
        if left_fit is not None and right_fit is not None:
            # Giới hạn nội suy từ nửa dưới ảnh (self.height // 2) trở xuống để tránh điểm dự đoán xa bị lệch ra ngoài đường
            ploty = np.linspace(self.height // 2, self.height - 1, self.height // 2)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            
            color_mask = np.zeros_like(segmented_img)
            cv2.fillPoly(color_mask, np.int_([pts]), (0, 255, 0))
            segmented_img = cv2.addWeighted(segmented_img, 1.0, color_mask, 0.4, 0)
            
            target_y = self.height - 20
            lx = left_fit[0]*target_y**2 + left_fit[1]*target_y + left_fit[2]
            rx = right_fit[0]*target_y**2 + right_fit[1]*target_y + right_fit[2]
            center_x = (lx + rx) / 2.0
            
            cv2.circle(segmented_img, (int(center_x), target_y), 6, (0, 0, 255), -1)

        return segmented_img, thresh, center_x, left_fit, right_fit
