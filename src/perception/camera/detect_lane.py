import cv2
import numpy as np

class LaneDetector:
    """
    Công cụ xử lý ảnh Computer Vision truyền thống để:
    1. Tìm biên trái (Left line) và biên phải (Right line) bằng thuật toán Sliding Window.
    2. Phân đoạn (Segmentation): Tô đa giác màu xanh lá cây vào khu vực "Drivable Area" (Trong lane).
    3. Trả về tọa độ Center để điều khiển xe.
    """
    def __init__(self, image_width=300, image_height=300):
        self.width = image_width
        self.height = image_height
        
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
