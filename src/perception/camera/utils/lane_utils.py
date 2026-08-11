import os
import cv2
import numpy as np

def init_espcn(sr_scale=2):
    """
    Khởi tạo đối tượng siêu độ phân giải ESPCN nếu được hỗ trợ. (Đã vô hiệu hóa)
    """
    print("[INFO] DNN Super Resolution đã bị vô hiệu hóa theo cấu hình.")
    return None

def enhance_image(frame, sr, sr_scale, width, height, detector):
    """
    Pipeline tăng cường chất lượng ảnh trước khi xử lý lane detection.
    Sử dụng Auto Gamma Correction (đã sửa công thức chuẩn) + CLAHE + Bilateral Filter.
    """
    if sr is not None:
        try:
            small_frame = cv2.resize(frame, (width // sr_scale, height // sr_scale))
            img = sr.upsample(small_frame)
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height))
        except Exception:
            img = cv2.resize(frame, (width, height))
    else:
        img = cv2.resize(frame, (width, height))

    # --- Lấy tham số an toàn từ detector ---
    gamma_target  = getattr(detector, '_gamma_target', 128)
    gamma_min     = getattr(detector, '_gamma_min', 0.4)
    gamma_max     = getattr(detector, '_gamma_max', 2.5)
    clahe_clip    = getattr(detector, '_clahe_clip', 2.5)
    clahe_grid    = getattr(detector, '_clahe_grid', 4)
    bilateral_d   = getattr(detector, '_bilateral_d', 5)
    bilateral_sc  = getattr(detector, '_bilateral_sc', 60)
    bilateral_ss  = getattr(detector, '_bilateral_ss', 60)

    # --- Auto Gamma Correction (Chuẩn hóa công thức tỷ lệ log) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mean = float(np.mean(gray))
    if 5 < gray_mean < 250:
        # Công thức chuẩn: gamma = log(mean / 255) / log(target / 255)
        # Giúp ảnh tối (mean thấp) -> gamma > 1.0 (sáng lên)
        # Giúp ảnh sáng (mean cao) -> gamma < 1.0 (tối đi)
        try:
            gamma = np.log(gray_mean / 255.0) / np.log(gamma_target / 255.0)
            gamma = float(np.clip(gamma, gamma_min, gamma_max))
            inv_gamma = 1.0 / gamma
            lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            img = cv2.LUT(img, lut)
        except Exception:
            pass

    # --- CLAHE trên kênh V (HSV) để tối ưu độ tương phản cục bộ ---
    try:
        hsv_temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        hsv_temp[:, :, 2] = clahe.apply(hsv_temp[:, :, 2])
        img = cv2.cvtColor(hsv_temp, cv2.COLOR_HSV2BGR)
    except Exception:
        pass

    # --- Bilateral Filter khử nhiễu hạt giữ nguyên cạnh của làn đường ---
    try:
        img = cv2.bilateralFilter(img, d=bilateral_d, sigmaColor=bilateral_sc, sigmaSpace=bilateral_ss)
    except Exception:
        pass

    return img

def detect_dashed_center(bev_hsv, red_mask, width, height, detector, boundary_fit=None, boundary_side='none'):
    """
    Phát hiện nét đứt trung tâm trên BEV dựa vào vị trí và hình dạng contour.
    """
    center_x_lo = int(width * detector._dash_center_lo)
    center_x_hi = int(width * detector._dash_center_hi)

    BOUNDARY_EXCLUSION = detector._dash_boundary_margin
    if boundary_fit is not None:
        ref_y = int(height * 0.7)
        x_bnd = float(boundary_fit[0]*ref_y**2 + boundary_fit[1]*ref_y + boundary_fit[2])
        x_bnd = float(np.clip(x_bnd, 0, width))
        if boundary_side == 'right':
            center_x_hi = min(center_x_hi, int(x_bnd) - BOUNDARY_EXCLUSION)
        elif boundary_side == 'left':
            center_x_lo = max(center_x_lo, int(x_bnd) + BOUNDARY_EXCLUSION)

    if center_x_lo >= center_x_hi - 20:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), False

    center_mask = np.zeros_like(red_mask)
    center_mask[:, center_x_lo:center_x_hi] = 255
    dash_mask = cv2.bitwise_and(red_mask, center_mask)

    cnts_data = cv2.findContours(dash_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts_data[0] if len(cnts_data) == 2 else cnts_data[1]

    valid_dashes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < detector._dash_area_min or area > detector._dash_area_max:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if h < detector._dash_h_min:
            continue

        aspect = w / float(h) if h > 0 else 99.0
        if aspect > detector._dash_aspect_max:
            continue

        M_c = cv2.moments(c)
        if M_c['m00'] > 0:
            cx_dash = M_c['m10'] / M_c['m00']
            cy_dash = M_c['m01'] / M_c['m00']
            valid_dashes.append((cx_dash, cy_dash, c))

    valid_dashes.sort(key=lambda d: d[1])

    dash_pts_x = np.array([], dtype=np.float32)
    dash_pts_y = np.array([], dtype=np.float32)
    dash_detected = False

    if len(valid_dashes) >= detector._dash_min_count:
        cx_vals = np.array([d[0] for d in valid_dashes])
        cx_median = np.median(cx_vals)
        aligned = [(d[0], d[1], d[2]) for d in valid_dashes
                   if abs(d[0] - cx_median) < detector._dash_align_tol]

        if len(aligned) >= detector._dash_min_count:
            all_pts = []
            for _, _, cnt in aligned:
                all_pts.append(cnt.reshape(-1, 2))
            all_pts = np.vstack(all_pts).astype(np.float32)
            dash_pts_x = all_pts[:, 0]
            dash_pts_y = all_pts[:, 1]
            dash_detected = True

    return dash_pts_x, dash_pts_y, dash_detected

def sliding_window_segment(thresh, resized, width, height, limit_y_ratio=0.0):
    """
    Áp dụng thuật toán Sliding Window trên ảnh nhị phân để dò lề đường
    và tô màu phân đoạn drivable area.
    limit_y_ratio: Tỷ lệ chiều cao bắt đầu nội suy đa giác (0.0: toàn bộ, 0.5: nửa dưới)
    """
    # 1. Tìm chân vạch đường bằng Histogram
    histogram = np.sum(thresh[height//2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    if histogram[leftx_base] < 50:
        leftx_base = 30
    if histogram[rightx_base] < 50:
        rightx_base = width - 30

    # 2. Sliding Window
    nwindows = 9
    window_height = int(height // nwindows)
    
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
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height

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

    # 3. Fit đa thức & Phân đoạn
    left_fit, right_fit = None, None
    segmented_img = resized.copy()

    if len(leftx) > 10:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 10:
        right_fit = np.polyfit(righty, rightx, 2)

    lane_width = 140.0
    if left_fit is not None and right_fit is None:
        right_fit = left_fit.copy()
        right_fit[2] += lane_width
    elif right_fit is not None and left_fit is None:
        left_fit = right_fit.copy()
        left_fit[2] -= lane_width

    center_x = width // 2

    if left_fit is not None and right_fit is not None:
        start_y = int(height * limit_y_ratio)
        ploty = np.linspace(start_y, height - 1, height - start_y)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        color_mask = np.zeros_like(segmented_img)
        cv2.fillPoly(color_mask, np.int_([pts]), (0, 255, 0))
        segmented_img = cv2.addWeighted(segmented_img, 1.0, color_mask, 0.4, 0)

        target_y = height - 20
        lx = left_fit[0]*target_y**2 + left_fit[1]*target_y + left_fit[2]
        rx = right_fit[0]*target_y**2 + right_fit[1]*target_y + right_fit[2]
        center_x = (lx + rx) / 2.0

        cv2.circle(segmented_img, (int(center_x), target_y), 6, (0, 0, 255), -1)

    return segmented_img, center_x, left_fit, right_fit
