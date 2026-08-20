# Smart City V2 — chạy giả lập trước, gắn AI sau

## Kết luận kiến trúc

AI **chỉ nhận diện ngữ nghĩa** của đèn và biển báo. AI không được điều khiển
servo hoặc motor.

```text
Camera ─┬─> OpenCV: làn, vạch dừng, vùng cấm xanh/cam ─┐
        └─> AI: biển + đèn + confidence ───────────────┤
                                                       v
Scenario/allowed exits ─> Decision rules ─> FSM ─> ga/lái ─> Safety ─> xe
```

Điều này cho phép dùng `scenario_example.json` khi model chưa có. Khi nhận
model, chỉ thay nguồn nhãn; thuật toán dừng, chọn hướng, cua và bắt lại làn giữ
nguyên.

## Những gì V2 đã xử lý

- Bám làn bằng các đoạn trắng gần/xa, không lấy tâm ảnh làm làn giả khi mất nét.
- Nhận vạch trước giao lộ bằng **nhiều thanh trắng cùng hàng**, rồi xác nhận qua
  nhiều frame mới; một nét đứt dọc hoặc một frame lóe sáng không đủ kích hoạt.
- Xem phần xanh và viền cam/đỏ là vùng keep-out để tạo cảnh báo và bias né.
- Dừng trước giao lộ, giữ xe đứng, đọc quyết định đúng một lần, sau đó thực thi
  primitive `LEFT`, `RIGHT` hoặc `STRAIGHT`.
- Ngã ba được mô tả bằng các exit thật sự tồn tại. Ví dụ phía trước là đảo xanh
  thì chỉ khai báo `LEFT`, `RIGHT`; biển `NO_LEFT` sẽ được giải thành `RIGHT`.
- Nhãn không biết, hướng bị cấm, route hết, camera cũ, mất làn, timeout cua hoặc
  vật cản quá gần đều dừng. Không có mặc định “đi thẳng” hay “rẽ phải”.
- Ảnh camera đóng băng không thể tự tích đủ debounce vì bộ đếm chỉ tăng với
  `frame_seq` mới.
- Chế độ mặc định là shadow: vẫn tính lệnh và vẽ debug nhưng không tạo driver
  motor.

FSM hiện dùng:

```text
DISARMED -> WAIT_SENSORS -> LANE_FOLLOW -> APPROACH_LINE
 -> STOP_HOLD -> WAIT_DECISION
 -> NUDGE/TURNING hoặc CROSSING -> REACQUIRE -> EXIT_LOCKOUT -> LANE_FOLLOW

SAFE_STOP, E_STOP_LATCHED và FINISHED luôn trả throttle = 0
```

## Khai báo đường giả lập

Sao chép
[`scenario_example.json`](../src/smart_city/v2/scenario_example.json), sau đó
điền giao lộ theo đúng thứ tự xe gặp từ vị trí xuất phát. Hướng luôn tính tương
đối với đầu xe khi đã dừng trước vạch.

```json
{
  "validated_for_live": false,
  "route_id": "REPLACE_WITH_MEASURED_ROUTE_ID",
  "preference": ["STRAIGHT", "RIGHT", "LEFT"],
  "intersections": [
    {
      "id": "I01_T",
      "allowed": ["LEFT", "RIGHT"],
      "mock_sign": "NO_LEFT"
    },
    {
      "id": "I02_4WAY",
      "allowed": ["LEFT", "STRAIGHT", "RIGHT"],
      "action": "STRAIGHT"
    },
    {
      "id": "FINISH",
      "allowed": ["STRAIGHT"],
      "action": "END"
    }
  ]
}
```

Quy tắc:

- `allowed` chỉ chứa đường không đi vào ô xanh/không ra ngoài sa hình.
- Chọn một trong `mock_sign` hoặc `action`, không dùng đồng thời.
- Biển hỗ trợ: `TURN_LEFT`, `TURN_RIGHT`, `GO_STRAIGHT`, `NO_LEFT`,
  `NO_RIGHT`, `NO_STRAIGHT`, `STOP`.
- `END` là đích rõ ràng. Hết mảng mà không gặp `END` là lỗi và xe dừng.
- File mẫu chỉ minh họa API, **không phải route chính thức**. Cần biết vị trí và
  hướng xuất phát mới điền được route thật.
- Chỉ sau khi đi bộ kiểm tra toàn tuyến mới đổi `validated_for_live` thành
  `true`; motor mode từ chối file mẫu và route chưa có `route_id`.

## Chạy an toàn theo từng tầng

Từ thư mục gốc dự án, kiểm thử logic thuần:

```bash
python3 -B -m unittest discover -s tests -p "test_smart_city_v2_*.py" -v
```

Kiểm tra một ảnh hoặc replay video, hoàn toàn không chạm motor:

```bash
python3 src/smart_city/main_smart_city_v2.py \
  --image anh_camera_tren_xe.jpg --display

python3 src/smart_city/main_smart_city_v2.py \
  --video smart_city_run.mp4 \
  --scenario src/smart_city/v2/scenario_example.json \
  --config src/smart_city/v2/config_example.json \
  --display --log smart_city_shadow.csv
```

ROS shadow trên Jetson: chương trình tự arm FSM để xem **lệnh dự kiến**, nhưng
không khởi tạo phần cứng.

```bash
python3 src/smart_city/main_smart_city_v2.py \
  --ros --use-lidar \
  --scenario src/smart_city/v2/scenario_example.json \
  --config src/smart_city/v2/config_example.json \
  --web-port 8080 --log smart_city_shadow.csv
```

Chỉ sau khi shadow replay đúng và đã nâng bánh khỏi mặt đất mới thử actuator:

```bash
python3 src/smart_city/main_smart_city_v2.py \
  --ros --use-lidar --enable-motors --arm \
  --scenario src/smart_city/v2/my_route.json \
  --config src/smart_city/v2/my_calibration.json \
  --log smart_city_live.csv
```

File calibration live phải có hai trường chủ ý sau (file mẫu để `false` nên sẽ
bị từ chối):

```json
{
  "calibrated": true,
  "calibration_id": "car01-course-2026-08-17"
}
```

Hai cờ `--enable-motors --arm` chỉ mở quyền arm. Sau khi camera/LiDAR đã ổn và
người giữ E-stop xác nhận khu vực trống, gửi một yêu cầu arm **mới**:

```bash
rosservice call /smart_city/arm
```

Arm dùng service one-shot nên lệnh `true` cũ/latched từ process trước không thể
tự kích hoạt lần chạy mới. Service chỉ nhận khi camera/LiDAR đang fresh và yêu
cầu tự hết hạn sau `arm_request_ttl_seconds` (mặc định 0,75 giây).

E-stop phần mềm độc lập với cửa sổ hiển thị:

```bash
rostopic pub -1 /smart_city/estop std_msgs/Bool "data: true"
```

E-stop này được latch; không gửi `false` để chạy tiếp. Nếu E-stop xảy ra giữa
giao lộ, phải đưa xe về mốc đã biết và restart route/process. Luôn có người giữ
E-stop vật lý. Watchdog actuator tự dừng và latch nếu không nhận heartbeat mới
trong 0,2 giây.

`throttle=0` chỉ là nhả ga, không phải phanh chủ động; cần đo quãng trôi theo
mức pin.

Trong cửa sổ debug:

- `q`/Esc: thoát; `e`: E-stop; `x`: reset; `a`: arm lại. Hai phím sau chỉ được
  bật trong shadow/offline; live phải dùng service hoặc relocalize/restart.
- `1`, `2`, `3`: giả lập đèn đỏ, vàng, xanh; `c`: xóa nhãn.
- `j`, `i`, `k`: `TURN_LEFT`, `GO_STRAIGHT`, `TURN_RIGHT`.
- `u`, `o`: `NO_LEFT`, `NO_RIGHT`.

## Hợp đồng với model AI

Cách an toàn nhất là chạy inference ở node/process riêng và publish JSON bằng
`std_msgs/String`, để inference chậm không chặn vòng điều khiển:

```json
{
  "sign_label": "NO_LEFT",
  "sign_confidence": 0.94,
  "signal_label": "GREEN",
  "signal_confidence": 0.98,
  "latency_ms": 47.2,
  "source_frame_seq": 1842,
  "source_stamp_ns": 1787049300123456789
}
```

Sau đó chạy:

```bash
python3 src/smart_city/main_smart_city_v2.py \
  --ros --semantic-topic /smart_city/semantic --require-ai \
  --scenario src/smart_city/v2/my_route.json \
  --config src/smart_city/v2/my_calibration.json
```

`--require-ai` làm xe giữ nguyên tại vạch nếu chưa có nhãn mới. Nhãn chạy phải
lặp ổn định ít nhất `ai_confirm_frames` message mới (mặc định 3); đỏ/vàng có thể
dừng ngay. Nhãn cũ quá
`semantic_ttl_seconds`, confidence dưới `ai_min_confidence`, đỏ/vàng hoặc nhãn
lạ đều không được cho xe chạy. Đèn xanh là “cổng cho đi”; hướng vẫn lấy từ biển
hoặc route hợp lệ. Đèn đỏ/vàng không consume bước route.

Mỗi kết quả live phải mang `source_frame_seq` và/hoặc `source_stamp_ns` lấy từ
`header` của đúng ảnh ROS đã inference. Runner chỉ nhận kết quả khớp một frame
camera gần đây, đúng thứ tự, chưa hết TTL và phát sinh sau lúc xe vào trạng thái
chờ quyết định. Vì vậy output AI trễ của giao lộ trước không thể dùng để rẽ ở
giao lộ sau. RED/YELLOW đến trước khi xe bắt đầu primitive sẽ giữ xe; sau khi đã
vào giao lộ, FSM tiếp tục thoát giao lộ thay vì phanh giữa đường, còn E-stop,
LiDAR và vùng keep-out vẫn có quyền dừng tức thời.

`FunctionSemanticDetector` trong `semantic.py` chỉ dành cho test offline/shadow.
Live phải chạy inference bất đồng bộ ở node riêng; nếu nhúng inference đồng bộ,
model chậm có thể làm watchdog dừng và latch xe. Adapter chỉ nhận bốn khóa nhãn
và confidence; kết quả chứa `steering`, `throttle`, `speed`, `motor`, `servo`
hoặc `action` sẽ bị từ chối.

Model cần gắn detection với ROI của hướng xe đang tiếp cận, không đơn giản lấy
bbox có confidence cao nhất toàn ảnh: biển/đèn được treo trên bốn góc cùng một
khung nhôm và camera có thể nhìn thấy tín hiệu của đường khác.

## Calibration bắt buộc trên camera gắn xe

Ảnh chụp toàn cảnh từ trên cao chỉ đủ hiểu topology; không dùng nó để chốt HSV,
ROI hay thời gian cua. Cần video/ảnh từ đúng camera CSI, đúng độ cao và góc lắp.

1. Khóa exposure/white balance, lấy mẫu HSV trắng, xanh và cam/đỏ ngay tại sân.
2. Vẽ overlay và chỉnh ROI để chân khung/biển phía trên không thành vạch dừng.
3. Đẩy xe bằng tay qua vạch, tìm `stop_line_y` phù hợp với cản trước và quãng trôi.
4. Tìm throttle nhỏ nhất xe chạy ổn định ở từng mức pin; đặt cruise chỉ cao hơn
   khoảng 0,02.
5. Nâng bánh, xác nhận dấu lái: mặc định trái âm, phải dương. Nếu phần cứng ngược
   thì đảo hai giá trị `turn_steering_left/right`.
6. Đo riêng `nudge_*_seconds`, góc lái và thời gian cua trái/phải trên sân.
7. Replay ít nhất một vòng bằng shadow, kiểm tra CSV: mỗi vạch chỉ consume một
   intersection, latency AI dưới 300 ms, không bão hòa lái kéo dài.
8. Hạ xe chạy từng primitive ở ga thấp và đặt chướng ngại mềm; không chạy full
   route ngay lần đầu.

Chân khung nhôm có thể tạo cluster LiDAR. Không tắt LiDAR chỉ để tránh false
positive; hãy xem log, chỉnh `lidar_guard_half_angle_deg` theo swept footprint.
Chân nằm trong quỹ đạo vẫn phải làm xe dừng.

V2 hiện dùng `allowed` trong scenario làm topology chuẩn; camera chưa chứng minh
độc lập từng exit trái/thẳng/phải bằng BEV swept-footprint. Vì vậy route sai thứ
tự vẫn là lỗi nghiêm trọng và motor mode chỉ là khung để calibration có giám
sát, chưa phải cấu hình “đặt xe xuống là chạy” từ hai ảnh toàn cảnh.
