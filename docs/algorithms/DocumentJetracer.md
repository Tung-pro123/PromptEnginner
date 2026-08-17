# Jetracer AI Racing - Hướng dẫn Tổng quan (Summer 2026)

Chào mừng các bạn đến với Jetracer AI Racing Summer 2026. 

Mục đích của Readme này là để các bạn tham khảo cách chia nhỏ các bài toán, cấu trúc chung và một số lời khuyên chứ không phải để copy y nguyên về cấu trúc thuật toán. Tùy thuộc vào sa bàn thực tế, các bạn sẽ phải tự tinh chỉnh thuật toán.

## 1. Môi trường & Phần mềm trên xe

Xe được cung cấp sẽ chạy với các phiên bản phần mềm sau:

| Môi trường | Phiên bản |
| :--- | :--- |
| Jetpack | 4.5.1 |
| CUDA | 10.2 |
| Python 3 | 3.6.9 |
| Python 2 | 2.7.1 |
| ROS 1 | Melodic |

**Lưu ý**: ROS Melodic dùng Python 2, trong khi các thư viện AI hiện nay dùng Python 3. Các bạn không thể import thư viện của ROS trực tiếp vào file AI viết bằng Python 3 mà cần dùng các giải pháp giao tiếp trung gian (như tạo node bridge). Trên các xe vẫn hỗ trợ docker nếu các bạn cần

## 3. ROS (Robot Operating System) là gì?

Nó giống như một mạng nội bộ trên xe. Thay vì gộp code camera, bánh xe, và AI vào một file khổng lồ dễ lỗi, ROS chia chúng thành các "Node" độc lập. Node Camera sẽ đẩy hình ảnh lên mạng, Node AI lấy hình ảnh đó về xử lý và đẩy ra lệnh lái, cuối cùng Node Động Cơ nhận lệnh lái để làm bánh xe quay.

## 2. Cấu trúc thư mục (Dựa theo Waveshare Jetracer ROS)
Github: https://github.com/waveshare/jetracer_ros


Đây là repo chính thức từ nhà sản xuất, với việc sử dụng các repo này sẽ giúp các bạn có thể lấy data các cảm biến trên xe nhanh hơn. Về các tài liệu và các script khác do nhà sản xuất đã code các bạn có thể tham khảo ở: https://www.waveshare.com/wiki/JetRacer_ROS_AI_Kit

- launch/: Chứa các file để khởi động nhiều bộ phận cùng lúc (ví dụ: bật camera, bật lidar, bật chế độ đi tự động).

- cfg/: Chứa file cấu hình để bạn có thể chỉnh thông số trực tiếp khi xe đang chạy mà không cần sửa code.

- config/: Chứa các file thông số cố định dạng YAML (như thông số bản đồ, khoảng cách né vật cản, cấu hình camera).

- scripts/: Chứa code viết bằng Python cho các tác vụ phụ trợ như xử lý âm thanh, lọc nhiễu Lidar, hay logic đi tuần tra nhiều điểm.

- src/: Chứa code C++ cấp thấp dùng để giao tiếp thẳng với động cơ và mạch điện của xe.

- maps/: Chứa các script giúp xe lưu lại bản đồ sau khi đi quét một vòng (SLAM).

## 3. Luồng hoạt động (Pipeline) gợi ý

Để xe tự chạy được, hệ thống nên có 3 khối chính:

- Perception: Lấy dữ liệu từ camera/Lidar để thấy làn đường, biển báo, vật cản.

- State machine: Xem xét nên rẽ, đi thẳng hay dừng lại khi nhận các dữ liệu từ Perception.

- Điều khiển: Gửi lệnh để động cơ quay bánh xe để xe di chuyển theo ý mình.

Lời khuyên: Hãy code theo kiểu xe đi từng bước (Dữ liệu từ xe lấy từ xa bàn -> Thay đổi các state machine/Xử lý -> Thực thi điều khiển xe -> Dừng xe đợi điều kiện tiếp theo). Đừng vội làm xe chạy liên tục từ đầu sẽ rất khó tìm ra lỗi.

4. Cách chạy thử cơ bản

Biên dịch code và nạp môi trường:
```
catkin_make
source devel/setup.bash
```

Bật các tính năng cơ bản của xe (động cơ, cảm biến):
```
roslaunch jetracer jetracer.launch
```

Hoặc bật toàn bộ hệ thống quét bản đồ và tự lái:
```
roslaunch jetracer slam_nav.launch
```

## 5. Lời khuyên khi viết code
Bước 1: Xác định bài toán cần giải quyết và tìm hiểu các phần cứng cần sử dụng trên xe.
Bước 2: Viết các luồng trên Jupyter Notebool/Python.

Bước 3: Chạy thử và điều chỉnh các thông số

Bước 4(Tùy vào các đội chọn): Port sang ROS.


## 6. Huấn luyện mô hình AI

Thu thập ảnh từ chính camera của xe.

Dùng máy tính cá nhân hoặc Google Colab để huấn luyện mô hình (không train trên xe vì sẽ rất chậm).

Nên dùng các mô hình nhẹ để xử lý nhanh do dù là đến từ NVIDIA nhưng Jetson Nano Orin vẫn là một GPU Edge AI, nó không quá mạnh như trên các máy tính thông thường.

**Tùy chọn**: Chuyển mô hình sang định dạng TensorRT của NVIDIA. Việc này giúp mô hình chạy mượt mà và nhận diện nhanh hơn hẳn trên bộ xử lý của xe.

**Lưu ý 1**: Các đội sử dụng phương pháp ROS cần căn nhắc ở điểm AI, do ROS1 sử dụng Python2, trong khi các model hiện nay thường sử dụng Python3, do đó cần tìm cách để triển khai trên ROS.

**Lưu ý 2**: Python3 mặc định của xe là 3.6.9, đây gần như là phiên bản tối đa mà nhà sản xuất hỗ trợ cho xe

## 7. Kết nối điều khiển xe

Máy tính của bạn và xe phải kết nối chung một mạng Wifi.
Để truy cập vào xe, mở Terminal và gõ:
```
ssh jetson@<ip_cua_xe>
```

**Các bạn cũng có thể truy cập xe qua Jupyter Notebook bằng cách**

Bước 1: Vào browser trên Laptop của bạn

Bước 2: Truy cập link ```<ip của jetson>:8888```

**Xe cũng có hỗ trợ NoMachine, có thể xem xét trên trang của nhà sản xuất để tìm hiểu thêm**

**Lưu ý:** Wifi của trường thường có tường lửa cản trở việc kết nối SSH. Khuyến khích các đội tự dùng điện thoại phát Wifi hoặc dùng laptop Wifi khi test xe. (Kể cả mạng Library vẫn bị các bạn nha).

## 8. Giới hạn phần cứng cần nhớ

Dù có GPU, bộ xử lý trên xe (Jetson) vẫn yếu hơn rất nhiều so với Laptop. Đừng kỳ vọng code chạy nhanh trên máy tính sẽ nhanh trên xe. Ngoài ra, xe dùng chung dung lượng RAM cho cả xử lý đồ họa và xử lý thường. Nếu load AI quá nặng, xe sẽ báo hết RAM và bị treo. Hãy liên tục tối ưu code!

### Một số tài liệu tham khảo:

Github ROS gốc của xe: https://github.com/waveshare/jetracer_ros

Wiki/Document của xe: https://www.waveshare.com/wiki/JetRacer_ROS_AI_Kit

Dataset mẫu các bạn có thể cân nhắc để hiểu cách label 
(Tác giả dataset: Goodgame và ICT teams của cuộc thi FPT Digital Race): 
https://via.makerviet.org/vi/models/traffic-sign-detection-yolov5/

Chúc các đội thi có một mùa giải thành công và học hỏi được nhiều kiến thức!