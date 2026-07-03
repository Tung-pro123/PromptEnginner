# 🏁 Cẩm Nang Cân Chỉnh Tham Số Speed Track (Calibration Guide)
*Tài liệu hướng dẫn tinh chỉnh tham số phần cứng thực tế cho đội PromptEngineer khi chạy xe trên sa bàn.*

---

## 📊 Bảng Tổng Hợp Các Tham Số Điều Chỉnh

Tất cả các tham số dưới đây nằm ở phần cấu hình đầu file [test_speed_track_concept.py](file:///d:/Jetson/Jetson/tests/test_spe| Tên tham số trong code | Tác dụng điều khiển | Giá trị mặc định | Hướng tinh chỉnh thực tế |
| :--- | :--- | :--- | :--- |
| **`BASE_SPEED`** | Tốc độ chạy cơ bản của xe (ga) | `0.22` | Tăng dần từ `0.12` (khi test) lên tối đa `0.40` (khi chạy thật) |
| **`Kp`** | Hệ số nhạy đánh lái bám làn | `0.007` | Giảm nếu xe bị lắc võng; Tăng nếu xe cua không hết góc |
| **`TRIGGER_DIST`** | Khoảng cách bắt đầu né (LiDAR trước) | `0.70` (mét) | Tăng nếu xe né trễ đâm hộp; Giảm nếu xe né quá sớm |
| **`DODGE_OFFSET_PX`** | Độ rộng đánh lái lách tránh vật cản | `70` (pixel) | Tăng giá trị này (lên `80` hoặc `90`) để bẻ cua né rộng hơn |
| **`MIN_DODGE_TIME`** | Thời gian tối thiểu giữ góc cua né | `2.0` (giây) | Tăng lên (ví dụ `2.5` hoặc `3.0` giây) để xe bẻ lái né lâu hơn, không trả lái sớm |
| **`RAMP_STEP_PX`** | Tốc độ bẻ lái chuyển làn (hình chữ S) | `5` (pixel) | Giảm nếu xe bẻ lái giật lật bánh; Tăng nếu xe phản xạ chậm |
| **`SIDE_CLEAR_DIST`** | Khoảng cách sườn trái an toàn trước khi nhập làn | `0.45` (mét) | Tăng lên nếu đuôi xe quẹt vào hộp khi quay lại làn cũ |
| **`thresh` value** | Ngưỡng lọc vạch trắng camera | `180` | Giảm nếu phòng thi tối; Tăng nếu sàn gỗ bị lóa ánh đèn |

---

## 🛠️ Hướng Dẫn Chi Tiết Cách Khắc Phục Lỗi Khi Chạy Xe

### 1. Hiện tượng xe đi lắc võng (hình con rắn) hoặc văng khỏi làn khi vào cua
* **Nguyên nhân:** Hệ số phản hồi lái `Kp` chưa khớp với cơ cấu lái vật lý của xe Ackermann.
* **Cách xử lý:**
  * **Nếu xe đi thẳng nhưng bánh trước lắc liên tục:** Bộ lái đang quá nhạy. Hãy giảm `Kp` xuống từng lượng nhỏ (ví dụ từ `0.007` $\rightarrow$ `0.005` $\rightarrow$ `0.004`).
  * **Nếu xe vào khúc cua nhưng không rẽ đủ, đâm thẳng ra ngoài biên:** Bộ lái đang quá lì. Hãy tăng `Kp` lên từ từ (ví dụ từ `0.007` $\rightarrow$ `0.009` $\rightarrow$ `0.011`).

### 2. Hiện tượng xe né vật cản chỉ bẻ lái 1 cái rồi trả lái ngay lập tức (Né hụt/Đâm thẳng)
* **Nguyên nhân:** Khi xe vừa chuyển sang trạng thái né (`STATE_DODGING`), vật cản vẫn còn ở phía trước chứ chưa sang bên hông xe. Do đó cảm biến bên hông trái (`is_left_side_clear()`) báo trống ngay lập tức, khiến xe bị lừa và chuyển sang trạng thái nhập làn (`STATE_REENTERING`) ngay trong frame tiếp theo.
* **Cách xử lý:**
  * **Tăng `MIN_DODGE_TIME`** (thời gian tối thiểu giữ cua né) lên thêm (ví dụ đặt là `2.0` hoặc `2.5` giây) để ép xe phải chạy lệch hẳn qua làn phải rồi mới được phép quét hông và quay đầu về.
  * **Tăng `DODGE_OFFSET_PX`** (độ lệch lách tránh) lên `80` hoặc `90` pixel để xe đánh góc lái rộng hẳn ra khi bắt đầu né.
  * *Mẹo: LiDAR đã được mở rộng quét từ 30 độ (hông trước) đến 110 độ để phát hiện vật cản chuyển dịch mượt hơn.*

### 3. Hiện tượng xe né vật cản quá trễ (đâm vào hộp) hoặc né quá sớm
* **Nguyên nhân:** Cự ly phản ứng cảm biến chưa phù hợp với tốc độ xe. Khi tốc độ ga (`BASE_SPEED`) tăng lên, quán tính xe lớn hơn nên xe cần nhận biết vật cản từ xa hơn.
* **Cách xử lý:**
  * **Nếu xe đâm vào hộp trước khi kịp lách:** Tăng `TRIGGER_DIST` lên (ví dụ lên `0.80` hoặc `0.90` mét).
  * **Nếu xe né sớm quá làm lệch quỹ đạo khi chưa tới gần hộp:** Giảm `TRIGGER_DIST` xuống (ví dụ xuống `0.55` hoặc `0.50` mét).

### 4. Hiện tượng xe cán vạch biên trắng bên phải khi đang né vật cản
* **Nguyên nhân:** Giá trị dịch chuyển làn ảo `DODGE_OFFSET_PX` quá lớn, ép xe phải lách quá rộng ra mép biên sa bàn.
* **Cách xử lý:**
  * **Nếu xe cán biên phải:** Giảm `DODGE_OFFSET_PX` xuống (ví dụ giảm từ `70` $\rightarrow$ `60` hoặc `50` pixel).
  * **Nếu xe lách quá hẹp và hông xe bên trái quẹt vào hộp:** Tăng `DODGE_OFFSET_PX` lên (ví dụ lên `80` hoặc `90` pixel).

### 5. Hiện tượng xe bẻ lái né quá đột ngột làm trượt bánh (mất lái)
* **Nguyên nhân:** Bước chuyển làn ảo `RAMP_STEP_PX` quá cao, khiến tâm đường ảo dịch chuyển đột ngột làm xe bẻ lái lái gắt.
* **Cách xử lý:**
  * **Nếu xe chuyển làn giật cục:** Giảm `RAMP_STEP_PX` xuống (ví dụ xuống `3` hoặc `4` pixel) để xe lướn chữ S mượt mà như tài xế lái.
  * **Nếu xe lượn quá lờ đờ không kịp tránh vật cản:** Tăng `RAMP_STEP_PX` lên (ví dụ lên `7` hoặc `8` pixel).

### 6. Hiện tượng đuôi xe quẹt vào hộp khi quay trở về làn cũ
* **Nguyên nhân:** Cảm biến LiDAR bên sườn xác nhận đã vượt qua vật cản quá sớm khi đuôi xe vẫn chưa qua khỏi hộp.
* **Cách xử lý:**
  * Tìm đến tham số `self.SIDE_CLEAR_DIST` trong code.
  * Tăng khoảng cách kiểm tra an toàn này lên thành `0.50` hoặc `0.60` mét để xe chạy lên xa hơn nữa mới đánh lái nhập làn.

### 7. Hiện tượng xe mất dấu đường (chạy loạn) khi thay đổi ánh sáng phòng thi
* **Nguyên nhân:** Ngưỡng nhị phân hóa (`thresh`) cố định không thích ứng được ánh sáng thực tế.
* **Cách xử lý:**
  * Tìm đến hàm `get_lane_centers()` trong code tại dòng `cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)`.
  * **Nếu sân thi tối:** Hạ ngưỡng từ `180` xuống `140` hoặc `150` để camera nhận diện được màu trắng mờ.
  * **Nếu sân thi bị lóa điện trần trên sàn gỗ:** Tăng ngưỡng lên `210` hoặc `220` để loại bỏ các đốm lóa phản chiếu.

---

## 📋 Quy Trình Cân Chỉnh Từng Bước Cho Đội Trên Sân Thi

Để tiết kiệm thời gian và tránh hỏng hóc xe, cả đội nên tuân thủ quy trình calibrate sau:

1. **Bước 1: Treo bánh xe (Kê cao xe)**
   * Kê khung gầm xe lên sao cho 4 bánh không chạm đất. Chạy lệnh test và dùng tay đặt vật cản trước LiDAR để xem bánh trước có tự đánh lái lách phải mượt mà không.
2. **Bước 2: Test bám làn với tốc độ cực thấp**
   * Đặt xe xuống sa bàn trống (không có vật cản). Cấu hình `BASE_SPEED = 0.12`. Chạy xe để tối ưu thông số bám làn thẳng `Kp` sao cho xe đi thẳng tắp ở giữa đường, không lắc bánh.
3. **Bước 3: Test né vật cản tốc độ thấp**
   * Đặt hộp carton lên đường. Cho xe chạy ở tốc độ `0.12` để tinh chỉnh cự ly né `TRIGGER_DIST` và độ rộng né `DODGE_OFFSET_PX`.
4. **Bước 4: Tăng tốc độ và tối ưu**
   * Tăng dần tốc độ ga `BASE_SPEED` lên mỗi lần `0.05` đơn vị. Mỗi lần tăng tốc, hãy tăng nhẹ `TRIGGER_DIST` tương ứng để bù đắp quán tính phanh/lái của xe.

---

## 📈 8. Hướng Dẫn Sử Dụng File CSV Để Phân Tích Lỗi (CSV Debugging)
Bên cạnh video, hệ thống hiện tự động lưu lại dữ liệu của mỗi lượt chạy vào file **`speed_track_debug.csv`** nằm tại thư mục chạy chương trình (`~/admin/Jetson/src/speed_track/speed_track_debug.csv`).

### Các cột thông tin trong file CSV:
* `timestamp`: Thời gian thực của hệ thống (giây).
* `state`: Trạng thái máy (FSM): `1` (Normal), `2` (Dodging), `3` (Reentering).
* `front_dist`: Khoảng cách tới vật cản trước mặt (mét).
* `closest_angle`: Góc của vật cản gần xe nhất ($-\mathbf{30}^\circ \rightarrow \mathbf{150}^\circ$).
* `closest_dist`: Khoảng cách tới điểm gần nhất của vật cản đó (mét).
* `clear_counter`: Bộ lọc lọc nhiễu trả lái (phải đạt đủ 8 mới thoát né).
* `current_offset_px`: Độ dịch chuyển làn ảo hiện tại (pixel).
* `steering`: Lệnh góc lái gửi xuống servo ($-1.0 \rightarrow 1.0$).

### 💡 Mẹo sử dụng file CSV để tinh chỉnh:
1. **Nếu xe né xong nhưng không chịu quay về làn (bị đâm biên phải):**
   - Mở file CSV, tìm vùng có cột `state` là `2`.
   - Xem giá trị cột `closest_dist` và `closest_angle`. Nếu thấy `closest_dist` luôn báo một con số nhỏ hơn `0.80` ở góc nào đó (ví dụ hông xe $90^\circ$ báo $0.75$m do bắt nhầm phải tường biên), điều này làm `clear_counter` không thể tăng lên 8.
   - **Giải pháp:** Hãy hạ nhẹ bán kính quét trong hàm `get_closest_obstacle_angle()` xuống `0.70m` hoặc `0.65m` để phớt lờ bức tường đó đi.
2. **Kiểm tra hoạt động của Watchdog Timer:**
   - Nếu sau 3.5 giây né mà xe vẫn bị kẹt ở trạng thái `2`, bạn sẽ thấy trong log hoặc file CSV dòng cảnh báo Watchdog kích hoạt. Xe sẽ tự trả `state` về `3` để cứu nguy. Bạn có thể giảm thời gian này xuống `2.5s` hoặc `3.0s` nếu sa bàn của bạn ngắn!
