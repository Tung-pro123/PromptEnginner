

---

## 📘 `SOLUTION.md`

```markdown
# 💡 Giải pháp điều khiển JetBot theo mô hình Event-Driven

## 🧠 Tư duy thiết kế

Mục tiêu của giải pháp là **điều khiển robot JetBot đi đến đích trong thời gian ngắn nhất bằng con đường tối ưu**, dựa trên bản đồ định sẵn và các tín hiệu môi trường như vạch đường và giao lộ.

---

## ⚙️ Kiến trúc tổng thể

1. **ROS Node (`jetbot_controller_node`)**: Quản lý vòng đời và trạng thái hoạt động của robot.
2. **Map Navigator**: Đọc `map.json` để xác định lộ trình tối ưu từ điểm xuất phát → đích.
3. **State Machine**: Quản lý luồng hoạt động theo sự kiện (LiDAR, hình ảnh, bản đồ).
4. **Computer Vision**: Phát hiện vạch đường và giao lộ bằng xử lý ảnh OpenCV.
5. **LiDAR Detector**: Phát hiện giao lộ chính xác và nhanh hơn thị giác.
6. **YOLO (tùy chọn)**: Nhận diện biển báo hoặc dữ liệu bổ sung nếu cần.
7. **MQTT Publisher**: Gửi dữ liệu xử lý ra ngoài để hiển thị hoặc ghi nhận.

---

## 📍 Thuật toán điều hướng

- Dự án sử dụng **thuật toán tìm đường ngắn nhất (Dijkstra hoặc BFS)** từ `MapNavigator`.
- Sau mỗi nút giao lộ, robot:
  1. Dừng lại.
  2. Kiểm tra node hiện tại có phải đích hay chưa.
  3. Nếu chưa, tra bản đồ để lấy hướng đi kế tiếp.
  4. Chuyển đổi hướng tuyệt đối sang hành động tương đối (`straight`, `left`, `right`).
  5. Thực thi hành động → tiếp tục hành trình.

---

## 🔄 Chu trình xử lý chính

1. **Chờ vạch đường xuất hiện**.
2. **Bắt đầu di chuyển theo vạch**.
3. **Phát hiện giao lộ** bằng LiDAR hoặc dự đoán từ ROI phía xa.
4. **Xử lý giao lộ**:
   - Dừng lại.
   - Cập nhật vị trí hiện tại.
   - Quyết định hướng đi tiếp theo theo bản đồ.
5. **Thực hiện rẽ và ổn định góc nhìn**.
6. **Bám lại vạch đường** và tiếp tục.
7. **Kết thúc khi đến đích**.

---

## 📊 Tối ưu hiệu năng

- 📸 **ROI kép**: sử dụng ROI chính và ROI dự báo để dự đoán mất vạch sớm hơn.
- 🚀 **Giới hạn lực bẻ lái** để di chuyển ổn định và mượt hơn.
- 🧭 **Cân nhắc thời gian xác nhận tín hiệu LiDAR** để tránh nhận nhầm tại node bắt đầu.
- 🔄 **Ghi video hành trình** để debug và đánh giá hiệu suất.

---

## 📌 Kết luận

Giải pháp kết hợp giữa **ROS**, **Computer Vision**, **LiDAR**, và **lập kế hoạch đường đi thông minh** để tạo ra một hệ thống điều khiển robot JetBot **tự động, tối ưu và có khả năng mở rộng**.

Hệ thống sẵn sàng để nâng cấp với các chức năng bổ sung như:
- Nhận diện biển báo để ưu tiên hướng đi.
- Giải bài toán từ QR code.
- Giao tiếp với hệ thống giám sát thời gian thực qua MQTT.

