# 🏁 Cẩm Nang Cân Chỉnh Tham Số Speed Track (Calibration Guide)
*Tài liệu hướng dẫn tinh chỉnh tham số phần cứng thực tế cho đội PromptEngineer khi chạy xe trên sa bàn.*

---

## 📊 Bảng Tổng Hợp Các Tham Số Điều Chỉnh

Tất cả các tham số dưới đây nằm ở phần cấu hình đầu file [test_speed_track_concept.py](file:///d:/Jetson/Jetson/tests/test_speed_track_concept.py) (từ dòng 31 đến 35) hoặc trong các hàm xử lý ảnh/LiDAR.

| Tên tham số trong code | Tác dụng điều khiển | Giá trị mặc định | Hướng tinh chỉnh thực tế |
| :--- | :--- | :--- | :--- |
| **`BASE_SPEED`** | Tốc độ chạy cơ bản của xe (ga) | `0.18` | Tăng dần từ `0.12` (khi test) lên tối đa `0.40` (khi chạy thật) |
| **`Kp`** | Hệ số nhạy đánh lái bám làn | `0.006` | Giảm nếu xe bị lắc võng; Tăng nếu xe cua không hết góc |
| **`TRIGGER_DIST`** | Khoảng cách bắt đầu né (LiDAR trước) | `0.65` (mét) | Tăng nếu xe né trễ đâm hộp; Giảm nếu xe né quá sớm |
| **`DODGE_OFFSET_PX`** | Độ rộng đánh lái lách tránh vật cản | `60` (pixel) | Giảm nếu xe cán biên phải; Tăng nếu xe quẹt vào hộp bên trái |
| **`RAMP_STEP_PX`** | Tốc độ bẻ lái chuyển làn (hình chữ S) | `5` (pixel) | Giảm nếu xe bẻ lái giật lật bánh; Tăng nếu xe phản xạ chậm |
| **`min_distances` threshold** | Khoảng cách check sườn trước khi nhập làn | `0.40` (mét) | Tăng lên nếu đuôi xe quẹt vào hộp khi quay lại làn cũ |
| **`thresh` value** | Ngưỡng lọc vạch trắng camera | `180` | Giảm nếu phòng thi tối; Tăng nếu sàn gỗ bị lóa ánh đèn |

---

## 🛠️ Hướng Dẫn Chi Tiết Cách Khắc Phục Lỗi Khi Chạy Xe

### 1. Hiện tượng xe đi lắc võng (hình con rắn) hoặc văng khỏi làn khi vào cua
* **Nguyên nhân:** Hệ số phản hồi lái `Kp` chưa khớp với cơ cấu lái vật lý của xe Ackermann.
* **Cách xử lý:**
  * **Nếu xe đi thẳng nhưng bánh trước lắc liên tục:** Bộ lái đang quá nhạy. Hãy giảm `Kp` xuống từng lượng nhỏ (ví dụ từ `0.006` $\rightarrow$ `0.005` $\rightarrow$ `0.004`).
  * **Nếu xe vào khúc cua nhưng không rẽ đủ, đâm thẳng ra ngoài biên:** Bộ lái đang quá lì. Hãy tăng `Kp` lên từ từ (ví dụ từ `0.006` $\rightarrow$ `0.008` $\rightarrow$ `0.010`).

### 2. Hiện tượng xe né vật cản quá trễ (đâm vào hộp) hoặc né quá sớm
* **Nguyên nhân:** Cự ly phản ứng cảm biến chưa phù hợp với tốc độ xe. Khi tốc độ ga (`BASE_SPEED`) tăng lên, quán tính xe lớn hơn nên xe cần nhận biết vật cản từ xa hơn.
* **Cách xử lý:**
  * **Nếu xe đâm vào hộp trước khi kịp lách:** Tăng `TRIGGER_DIST` lên (ví dụ lên `0.80` hoặc `0.90` mét).
  * **Nếu xe né sớm quá làm lệch quỹ đạo khi chưa tới gần hộp:** Giảm `TRIGGER_DIST` xuống (ví dụ xuống `0.55` hoặc `0.50` mét).

### 3. Hiện tượng xe cán vạch biên trắng bên phải khi đang né vật cản
* **Nguyên nhân:** Giá trị dịch chuyển làn ảo `DODGE_OFFSET_PX` quá lớn, ép xe phải lách quá rộng ra mép biên sa bàn.
* **Cách xử lý:**
  * **Nếu xe cán biên phải:** Giảm `DODGE_OFFSET_PX` xuống (ví dụ giảm từ `60` $\rightarrow$ `50` hoặc `45` pixel).
  * **Nếu xe lách quá hẹp và hông xe bên trái quẹt vào hộp:** Tăng `DODGE_OFFSET_PX` lên (ví dụ lên `70` hoặc `75` pixel).

### 4. Hiện tượng xe bẻ lái né quá đột ngột làm trượt bánh (mất lái)
* **Nguyên nhân:** Bước chuyển làn ảo `RAMP_STEP_PX` quá cao, khiến tâm đường ảo dịch chuyển đột ngột làm xe bẻ lái lái gắt.
* **Cách xử lý:**
  * **Nếu xe chuyển làn giật cục:** Giảm `RAMP_STEP_PX` xuống (ví dụ xuống `3` hoặc `4` pixel) để xe lướn chữ S mượt mà như tài xế lái.
  * **Nếu xe lượn quá lờ đờ không kịp tránh vật cản:** Tăng `RAMP_STEP_PX` lên (ví dụ lên `7` hoặc `8` pixel).

### 5. Hiện tượng đuôi xe quẹt vào hộp khi quay trở về làn cũ
* **Nguyên nhân:** Cảm biến LiDAR bên sườn xác nhận đã vượt qua vật cản quá sớm khi đuôi xe vẫn chưa qua khỏi hộp.
* **Cách xử lý:**
  * Tìm đến hàm `is_side_clear()` trong code.
  * Tăng khoảng cách kiểm tra an toàn tại dòng `return min(side_distances) > 0.40` lên thành `0.50` hoặc `0.60` mét để xe chạy lên thêm một đoạn ngắn nữa mới đánh lái nhập làn.

### 6. Hiện tượng xe mất dấu đường (chạy loạn) khi thay đổi ánh sáng phòng thi
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
