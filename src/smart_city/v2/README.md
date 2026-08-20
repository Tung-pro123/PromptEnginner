# Smart City V2 — README bàn giao cho Antigravity

Cập nhật: 2026-08-21

## 1. Mục tiêu và phạm vi

Module này được tạo cho bài thi Smart City (bài 2). Thiết kế hiện tại cố ý tách
AI ra khỏi điều khiển xe:

- AI chỉ nhận diện **biển báo**, **màu đèn giao thông**, confidence và định danh
  frame nguồn (`source_frame_seq`, `source_stamp_ns`).
- AI không được xuất steering, throttle, speed, action, exit hay crosswalk.
- OpenCV nhận biết làn trắng, một hoặc nhiều cụm zebra marker của giao lộ, vùng
  xanh và viền cam/đỏ.
- FSM quyết định lúc bám làn, dừng, chờ, rẽ, đi thẳng và bắt lại làn.
- File scenario mô phỏng thứ tự giao lộ khi model AI chưa có.
- Mọi đầu vào thiếu, cũ, sai kiểu, không chắc chắn hoặc trái topology đều phải
  làm xe đứng yên/fail-closed, không tự đoán hướng.

Ảnh toàn cảnh sân cho thấy đường màu đen, các đảo xanh là vùng cấm, viền đảo và
mép sân có màu cam/đỏ, trước giao lộ có một hoặc nhiều cụm thanh trắng. Đèn và biển
đặt trên khung nhôm bốn chân ở các góc. Ảnh toàn cảnh chỉ đủ hiểu topology;
không đủ để chốt HSV, ROI, góc lái hay thời gian cua của camera gắn trên xe.

## 2. Không dùng entrypoint cũ

Entrypoint mới:

```text
src/smart_city/main_smart_city_v2.py
```

Không tiếp tục phát triển `src/smart_city/main_smart_city.py` cho luồng V2. File
cũ từng gọi API điều khiển không tồn tại và không có các lớp safety mới.

## 3. Kiến trúc

```text
Camera ROS/video/image
        |
        v
SmartCityPerception ------------------------------+
  - lane near/far                                  |
  - 1..N zebra-marker groups                       |
  - green + orange/red keep-out                    v
                                           SmartCityFSM
AI node --> sign/light/confidence/source ID ------>  |
Scenario JSON --> allowed exits/mock action ------>  |
LiDAR -------------------------------------------->  |
                                                    v
                                           DriveCommand
                                                    |
                                      Safety supervisor/watchdog
                                                    |
                                                    v
                                               Actuator
```

Luồng trạng thái chính:

```text
DISARMED
  -> WAIT_SENSORS
  -> LANE_FOLLOW
  -> APPROACH_LINE
  -> STOP_HOLD
  -> WAIT_DECISION
  -> NUDGE -> TURNING -> REACQUIRE_CENTER
       hoặc CROSSING -> REACQUIRE_CENTER
  -> EXIT_LOCKOUT
  -> LANE_FOLLOW
```

Các trạng thái chặn:

```text
FINISHED, SAFE_STOP, E_STOP_LATCHED
```

## 4. Các file đã tạo

```text
src/smart_city/main_smart_city_v2.py       runner offline/ROS, actuator, watchdog
src/smart_city/v2/config.py                cấu hình và kiểm tra live hard limits
src/smart_city/v2/config_example.json      cấu hình mẫu, cố ý calibrated=false
src/smart_city/v2/perception.py            lane/stop-line/keep-out bằng OpenCV
src/smart_city/v2/decision.py              route tuần tự và giải biển báo
src/smart_city/v2/scenario_example.json    route minh họa, không phải route thi
src/smart_city/v2/semantic.py              hợp đồng dữ liệu AI
src/smart_city/v2/controller.py            FSM fail-closed
docs/SMART_CITY_V2_GUIDE.md                hướng dẫn vận hành/calibration chi tiết
tests/test_smart_city_v2_*.py              unit/regression tests
```

Các file này được cô lập trên nhánh `recovery/smart-city-v2`, xuất phát từ
checkpoint phục hồi `18e36cc`. Không xóa/reset các thay đổi khác của chủ dự án
trong `speed_track`, `core`, `config`, archive hoặc file không liên quan.

## 5. Những phần đã làm

### Perception

- Resize frame theo config rồi segment màu trắng, xanh và cam/đỏ trong HSV.
- Tìm cặp marker trái/phải ở dải gần và xa để suy ra tâm làn.
- Nếu không có cặp biên gần hợp lệ thì trả `lane_x_near=None`, không lấy tâm ảnh
  làm làn giả.
- Mỗi zebra marker phải gồm nhiều blob trắng tách rời, cùng hàng và đủ span
  ngang.
- Một nét đứt dọc hoặc lóe trắng một frame không đủ kích hoạt giao lộ.
- Một hoặc nhiều cụm marker trước cùng giao lộ vẫn thuộc một intersection
  episode; chỉ marker mới sau `EXIT_LOCKOUT` mới được tạo episode tiếp theo.
- Khi nhiều cụm cùng xuất hiện, cụm xa hơn theo hướng chạy làm gate proxy;
  `NUDGE` phải thấy marker clear ổn định hoặc cue keep-out của cổng rồi mới cua,
  đồng thời có hard timeout để không bò mù vô hạn.
- Tính mật độ keep-out ở trước/trái/phải và bias tránh vùng xanh.

### Decision/scenario

- Hỗ trợ `LEFT`, `STRAIGHT`, `RIGHT`, `STOP`, `END`.
- Hỗ trợ biển lệnh `TURN_LEFT`, `TURN_RIGHT`, `TURN_STRAIGHT`, `GO_STRAIGHT`.
- Hỗ trợ biển cấm `NO_LEFT`, `NO_RIGHT`, `NO_STRAIGHT`.
- Hướng cuối luôn phải nằm trong `allowed` của giao lộ hiện tại.
- Đèn đỏ/vàng giữ xe và không consume bước route.
- Đèn xanh chỉ là cổng cho đi; hướng vẫn lấy từ biển hoặc route hợp lệ.
- Route hết mà không có `END` thì dừng lỗi, không mặc định đi thẳng.
- Một intersection episode chỉ consume đúng một route entry; khóa route và
  `EXIT_LOCKOUT` chống các cụm marker kế tiếp đọc route lần nữa.

### FSM/control

- Debounce chỉ đếm frame camera mới, không đếm số vòng lặp.
- Vạch dừng cần nhiều frame và chuyển động y tương đối nhất quán.
- Stop hold dùng thời gian monotonic, không dùng `sleep` trong FSM.
- Rẽ dùng primitive `NUDGE/TURN/REACQUIRE_CENTER` có timeout.
- `REACQUIRE_CENTER` cần cả sai lệch ngang và heading nằm trong ngưỡng qua nhiều
  frame mới trước khi vào `EXIT_LOCKOUT`.
- Mất làn khi đang follow/approach/exit làm throttle về 0 ngay, sau đó latch
  SAFE_STOP nếu kéo dài.
- Camera/LiDAR stale, distance không finite, command NaN/Inf và timeout đều
  fail-closed.
- Nếu dừng giữa giao lộ sau khi route đã consume thì không cho reset rồi chạy
  tiếp mù; phải đưa xe về mốc đã biết và gọi recovery route rõ ràng.

### Runner và safety

- Mặc định là shadow, không tạo driver motor.
- Motor chỉ được phép với `--ros --use-lidar --enable-motors --arm`, một
  `--semantic-topic` thật và `--require-ai`; mọi mock label bị từ chối. `--arm`
  chỉ bật chính sách/service one-shot, **không tự arm và không tự chạy**.
- Live còn bắt buộc config riêng có `calibrated=true`, `calibration_id`, scenario
  riêng có `validated_for_live=true`, `route_id`; file mẫu bị từ chối.
- Arm live là service one-shot `/smart_city/arm`, chỉ nhận khi camera/LiDAR mới,
  và request có TTL ngắn.
- E-stop ROS `/smart_city/estop` được latch.
- Actuator có watchdog thread độc lập, hard cap throttle/steering và từ chối
  hardware mock.
- E-stop mới nhất đã được latch ngay trong actuator để đóng race: nếu callback
  E-stop tới giữa lúc FSM tính lệnh và actuator apply, lệnh chờ không thể bật ga
  trở lại.
- Camera callback từ chối header trùng/đi lùi và pixel CRC trùng liên tiếp.
- LiDAR chuẩn hóa góc cả kiểu `[-pi, pi]` lẫn `[0, 2pi]` và có yaw offset.
- Loop ROS dùng wall-clock sleep để `/use_sim_time` bị đứng không giữ ga vô hạn.

### Biên AI mới nhất

- Runtime luôn ghép đúng semantic observation với semantic sequence của cùng
  snapshot; không dùng label cũ với sequence mới giữa hai frame camera.
- Nhãn rỗng, sai kiểu, confidence thiếu/thấp hoặc hai màu đèn mâu thuẫn chỉ giữ
  xe đứng yên, không fallback vào route giả lập.
- Khi đã thấy AI tại một event, mất AI giữa debounce không được rơi về scenario.
- Kết quả AI live bắt buộc chỉ ra frame nguồn và phải khớp history camera gần
  đây, đúng thứ tự, chưa quá TTL.
- Kết quả có frame nguồn trước khi vào `WAIT_DECISION` không được dùng cho giao
  lộ hiện tại.

## 6. Hợp đồng AI

AI nên chạy ở node/process riêng và publish `std_msgs/String` chứa JSON:

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

Quy tắc:

- `source_frame_seq` lấy từ `sensor_msgs/Image.header.seq`.
- `source_stamp_ns` lấy từ `Image.header.stamp.to_nsec()`.
- Nên gửi cả hai. Nếu camera bridge để header bằng 0, runner còn hỗ trợ
  `source_crc32`, nhưng nên sửa bridge/publisher để có header thật.
- Mỗi frame nguồn chỉ publish một kết quả cuối; không lặp cùng ID để đủ debounce.
- Các ID phải tăng theo thời gian. Kết quả out-of-order bị bỏ.
- Không được có các khóa `steering`, `throttle`, `speed`, `motor`, `servo`,
  `action`, `exit`, `crosswalk` hay output geometry.
- Mỗi hướng tiếp cận chỉ có một mặt đèn quay về phía xe. Vì vậy nên crop ROI cố
  định đã calibrate và phân loại màu; bounding box không bắt buộc trong contract
  (có thể giữ riêng để debug).
- Live luôn bắt buộc `--require-ai`: chỉ có biển mà thiếu kênh đèn vẫn phải đứng;
  cần GREEN mới, đủ confidence/debounce và đúng source frame mới được đi.

`models/best.pt` đang có là YOLO lane/crosswalk, **không phải** model semantic
biển/đèn và không được đưa vào giao diện AI nói trên.

Policy đèn đỏ/vàng: tín hiệu nhận được trong `WAIT_DECISION` hoặc lúc xe mới
`NUDGE` tới cổng sẽ giữ xe. Sau lần giữ này phải xác nhận lại đủ số message
GREEN mới được tiếp tục; thời gian đứng chờ không bị tính vào timeout chuyển
động. Khi xe đã vào `TURNING`/`CROSSING`, FSM ưu tiên thoát giao lộ thay vì
phanh giữa đường; lúc đó E-stop, LiDAR và keep-out vẫn có quyền dừng tức thời.

## 7. Format scenario

Ví dụ tối thiểu:

```json
{
  "version": 1,
  "validated_for_live": false,
  "route_id": "DRY_RUN_ONLY",
  "preference": ["STRAIGHT", "RIGHT", "LEFT"],
  "intersections": [
    {
      "id": "I01_T",
      "allowed": ["LEFT", "RIGHT"],
      "requires_sign": true,
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

`allowed` không phải output AI. Đây là topology phải đi bộ kiểm tra thủ công:
chỉ liệt kê các exit không vào đảo xanh và không ra khỏi sa hình. Không đổi
`validated_for_live` thành true cho tới khi đã xác nhận đúng vị trí xuất phát,
hướng đầu xe và toàn bộ thứ tự giao lộ.

`mock_sign`/`mock_signal` chỉ dùng cho offline và shadow. Validation live từ
chối mọi mock nằm cả trong CLI lẫn scenario; route thi dùng `action` làm phương
án hình học, còn biển/đèn thật phải đến qua semantic topic.

Đặt `requires_sign: true` cho mọi giao lộ mà biển là dữ kiện bắt buộc (đặc biệt
ngã ba dùng biển cấm để chọn lối còn lại). Ở event đó, chỉ GREEN mà thiếu biển,
biển lạ hoặc biển dẫn tới hướng không thuộc `allowed` đều giữ xe và không
consume route. Không đặt cờ này cho giao lộ thực sự không có biển.

Trong offline/shadow, `mock_sign` được phép thay cho biển khi AI hoàn toàn
vắng mặt để route mẫu vẫn replay được. Một khi AI đã trả nhãn, nhãn lạ hoặc
không an toàn không được fallback về mock. Live validation luôn cấm mọi mock.

## 8. Năm tầng kiểm thử bắt buộc

Không dùng một con số test pass cũ làm chứng nhận live. Sau mỗi patch phải chạy
lại đúng commit/config sẽ đưa lên xe và đi tuần tự qua năm tầng:

1. Unit/regression với frame tổng hợp; tất cả test phải pass.
2. Một ảnh rồi replay video CSI ở shadow mode, kiểm tra overlay và CSV.
3. ROS shadow với camera, semantic node và LiDAR thật nhưng không tạo actuator.
4. Actuator khi bánh nâng khỏi mặt đất: polarity, arm service, watchdog, stale
   sensor và E-stop.
5. Sân trống ga thấp: từng primitive, từng giao lộ, cuối cùng mới full route.

Không được nhảy tầng nếu tầng trước còn failure hoặc output chưa giải thích được.

### Lệnh test đúng

Chạy từ root repository:

```powershell
python -B -m unittest discover -s tests -p "test_smart_city_v2_*.py" -v
```

Không dùng `python -m unittest tests.test_smart_city_v2_runner` vì thư mục
`tests` hiện không phải package. Muốn chạy riêng runner test:

```powershell
python -B -m unittest discover -s tests -p "test_smart_city_v2_runner.py" -v
```

Kiểm tra grammar Python 3.6:

```powershell
python -B -c "import ast,pathlib; files=list(pathlib.Path('src/smart_city/v2').glob('*.py'))+[pathlib.Path('src/smart_city/main_smart_city_v2.py')]+list(pathlib.Path('tests').glob('test_smart_city_v2_*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p), feature_version=(3,6)) for p in files]; print('Python 3.6 grammar OK:', len(files), 'files')"
```

Smoke test một ảnh:

```powershell
python -B src\smart_city\main_smart_city_v2.py --image only_camera_test.jpg --max-frames 1
```

Xác minh cuối trên máy phát triển ngày 2026-08-21: `98/98` Smart City V2 tests
pass, 12 file parse được với grammar Python 3.6, `git diff --check` sạch và smoke
một ảnh hoàn tất với exit code 0. Đây là xác minh logic/offline, không thay cho
ROS/hardware/calibration; phải chạy lại chính các lệnh này trên commit và config
sẽ đưa lên Jetson.

Replay video shadow:

```powershell
python -B src\smart_city\main_smart_city_v2.py --video VIDEO_CSI.mp4 --scenario src\smart_city\v2\scenario_example.json --display --log smart_city_shadow.csv
```

## 9. Cách check kết quả shadow

Overlay cần xem:

- lane near/far có bám đúng cặp marker của làn xe hay bắt nhầm hàng ngang;
- zebra marker chỉ bật ở hàng nhiều thanh trắng trước giao lộ;
- `stop_line_y` tiến gần camera tương đối đều; chỉnh
  `stop_approach_y_ratio` và `stop_close_y_ratio` theo camera/cản trước thật;
- mask xanh/cam phủ đảo cấm nhưng không phủ mặt đường đen;
- frame đứng hình phải dẫn tới camera stale, không tự đủ debounce;
- một hoặc nhiều cụm marker trong cùng episode chỉ consume một route entry;
- đỏ/vàng giữ `throttle=0`, xanh cần đủ confirmation;
- hướng quyết định luôn thuộc `allowed`;
- mất làn/keep-out/timeout đều về 0, không giữ lệnh ga cũ.

CSV cần kiểm tra:

- không có throttle dương trong `DISARMED`, `WAIT_SENSORS`, `STOP_HOLD`,
  `WAIT_DECISION`, `SAFE_STOP`, `E_STOP_LATCHED`;
- không có NaN/Inf ở command;
- steering không bão hòa kéo dài;
- loop latency thấp hơn actuator watchdog;
- camera age và AI latency nằm trong TTL;
- intersection ID/action đi đúng thứ tự route;
- không consume hai entry trước khi qua `REACQUIRE_CENTER` và `EXIT_LOCKOUT`.

## 10. Quy trình calibration bắt buộc

1. Chốt vị trí xuất phát, hướng đầu xe và thứ tự mọi giao lộ.
2. Quay ít nhất 20–30 giây bằng đúng camera CSI, đúng độ cao/góc/exposure, gồm
   bám làn, tiến gần một vạch, ngã T, ngã tư, đảo xanh và khung nhôm.
3. Chạy shadow, lưu video + CSV + overlay.
4. Lấy HSV trắng/xanh/cam-đỏ từ video onboard, không lấy trực tiếp ảnh toàn cảnh.
5. Chỉnh ROI để chân khung nhôm/biển phía trên không thành vạch dừng.
6. Xác nhận zebra marker 3+ frame; một hoặc nhiều cụm trước cùng giao lộ chỉ tạo
   một episode và route step chỉ consume một lần.
7. Nâng bánh khỏi mặt đất; đo polarity steering trái/phải, throttle deadzone,
   watchdog và E-stop.
8. Chạy đường thẳng ga thấp; đo quãng trôi vì throttle 0 chỉ nhả ga, không phải
   phanh chủ động.
9. Chạy từng primitive trái/phải/thẳng riêng với vật cản mềm; đo nudge, turn
   nominal/max và thời điểm bắt lại làn.
10. Hiệu chỉnh LiDAR yaw, khoảng dừng và sector theo kích thước xe/khung nhôm.
11. Đi bộ xác nhận `allowed` ở từng event rồi tạo scenario thật.
12. Chỉ sau nhiều vòng shadow đúng mới tạo config live riêng và ký
    `calibrated=true` + `calibration_id`.

## 11. Chạy ROS an toàn

Shadow trước, không motor:

```bash
python3 src/smart_city/main_smart_city_v2.py \
  --ros --use-lidar \
  --scenario src/smart_city/v2/scenario_example.json \
  --display --log smart_city_shadow.csv
```

Live chỉ sau calibration, nâng bánh trước:

```bash
python3 src/smart_city/main_smart_city_v2.py \
  --ros --use-lidar --enable-motors --arm \
  --scenario src/smart_city/v2/my_route.json \
  --config src/smart_city/v2/my_calibration.json \
  --semantic-topic /smart_city/semantic --require-ai \
  --log smart_city_live.csv
```

Sau khi camera/LiDAR đã fresh:

```bash
rosservice call /smart_city/arm
```

`--arm` trong lệnh khởi động chỉ enable service/policy trên; nó không auto-arm.
Live còn bắt buộc config đã đo có `calibrated=true` và `calibration_id`, scenario
route thật có `validated_for_live=true` và `route_id`, LiDAR fresh, actuator
watchdog hoạt động và người giữ E-stop vật lý.

E-stop ROS:

```bash
rostopic pub -1 /smart_city/estop std_msgs/Bool "data: true"
```

Luôn phải có E-stop vật lý/deadman độc lập; software watchdog không bảo vệ được
trường hợp process chết, kernel treo, GIL/hardware setter khóa vĩnh viễn hoặc
nguồn/mạch công suất lỗi.

### Checklist một ngày cuối

1. Chạy full test/grammar và tạo checkpoint Git phục hồi được.
2. Chốt vị trí, hướng xuất phát, `allowed` và route thật bằng cách đi bộ sa hình.
3. Quay video CSI; calibrate HSV/ROI marker và keep-out xanh/đỏ bằng shadow.
4. Cắm AI sign/light đúng source identity hoặc dùng mock để test FSM; không dùng
   `models/best.pt` làm model semantic.
5. Kiểm tra episode 1/N marker, route consume một lần và chuỗi
   `TURN -> REACQUIRE_CENTER -> EXIT_LOCKOUT`.
6. Nâng bánh thử arm, polarity, stale camera/LiDAR, watchdog và hai lớp E-stop.
7. Chạy ga thấp từng primitive với vật cản mềm; không chắc chắn thì dừng.

## 12. Việc còn thiếu / giới hạn đã biết

### P0 — chưa được phép chạy thi thật

- Chưa có video camera CSI chính thức và chưa calibrate HSV/ROI.
- Chưa biết vị trí/hướng xuất phát và sequence route chính thức.
- Chưa thử ROS thật, camera driver thật, LiDAR thật hay RacerController trên xe.
- Chưa đo polarity/deadzone/quãng trôi/góc cua theo pin và mặt sân.
- Chưa có E-stop vật lý/deadman.
- `allowed` hiện là topology nhập tay; route mẫu không phải route thi.
- MVP hiện **chưa live-ready**; không bật motor chỉ vì unit test đã pass.

### P0/P1 — geometry cần phát triển tiếp

- Keep-out hiện có guard ảnh phía trước và guard nửa ảnh theo hướng cua, nhưng
  chưa có BEV metric/swept footprint thật cho góc ngoài bánh trước và góc trong
  bánh sau. `turn_green_hard_stop_ratio` cùng margin vẫn phải calibrate theo
  kích thước/quỹ đạo thật; đây là giới hạn quan trọng cạnh đảo xanh.
- LiDAR đang là sector phía trước, chưa phải swept sector theo steering/action.
- Turn completion vẫn dựa thời gian + bắt lại làn; nên thêm IMU/yaw target hoặc
  visual heading đáng tin cậy để tránh bắt lại làn cũ/quay quá góc.
- Viền cam/đỏ mảnh và vinyl bóng có thể cần morphology/perspective riêng.
- Chân khung nhôm có thể là obstacle thật hoặc false positive; không được tắt
  toàn bộ LiDAR để né false positive.

### P1 — tích hợp AI

- Node AI phải publish source header theo contract mới.
- Cần test delayed/out-of-order/duplicate inference trên ROS thật.
- Cần crop ROI cố định bằng ảnh onboard cho mặt đèn duy nhất quay về phía xe;
  bbox không bắt buộc.
- Cần log confusion matrix theo lớp biển/đèn và điều kiện sáng thật.
- `models/best.pt` chỉ nhận lane/crosswalk, không thay cho model semantic này.

## 13. Quy trình sửa chữa liên tục

Mỗi lỗi phải đi theo vòng lặp này:

1. Ghi lại input gây lỗi: video/frame, config, scenario, CSV và state/reason.
2. Tạo unit/regression test tái hiện lỗi trước khi sửa.
3. Chỉ sửa đúng layer sở hữu lỗi:
   - mask/geometry ở `perception.py`;
   - luật biển/topology ở `decision.py`;
   - transition/timeout ở `controller.py`;
   - ROS/watchdog/hardware boundary ở runner;
   - range/hard cap ở `config.py`.
4. Chạy test file liên quan.
5. Chạy toàn bộ `test_smart_city_v2_*.py`.
6. Chạy grammar Python 3.6.
7. Replay lại cùng video và so CSV trước/sau.
8. Replay thêm video âm tính để tránh sửa một case nhưng phá case khác.
9. Chỉ thử xe sau khi shadow không phát lệnh nguy hiểm.
10. Thay đổi một nhóm tham số nhỏ mỗi lần và ghi `calibration_id`/ghi chú.

Không sửa lỗi bằng cách:

- hạ debounce xuống một frame;
- tăng timeout rất lớn;
- cho mất làn vẫn chạy ga tối thiểu;
- coi UNKNOWN là đường trống;
- mặc định đi thẳng/phải khi thiếu quyết định;
- tắt keep-out/LiDAR để tránh false positive;
- cho AI điều khiển motor trực tiếp.

## 14. Checklist bàn giao cho Antigravity

1. Đọc file này và `docs/SMART_CITY_V2_GUIDE.md`.
2. Chỉ làm trong các file Smart City V2/tests liên quan; giữ nguyên dirty worktree
   ngoài phạm vi.
3. Chạy full tests theo lệnh discover; không tiếp tục nếu còn bất kỳ failure nào.
4. Rà riêng các sửa đổi an toàn cuối:
   - actuator E-stop atomic latch;
   - Runtime semantic observation/sequence pairing và TTL tính từ frame nguồn;
   - RosBuffers camera-source matching và semantic monotonicity;
   - `requires_sign` không fallback action ở giao lộ bắt buộc đọc biển;
   - planned-side keep-out ngay tại frame `NUDGE -> TURNING`.
5. Thêm/điều chỉnh regression nếu test giả ROS có lỗi import/API.
6. Chạy Python 3.6 grammar và offline image smoke.
7. Không bật motor với config/scenario mẫu.
8. Xin người dùng video CSI, vị trí/hướng xuất phát, route và thông số xe trước
   khi hiệu chỉnh.
9. Ưu tiên BEV swept-footprint + yaw/heading feedback trước live full course.
10. Báo cáo rõ phần đã xác minh bằng laptop, phần chỉ shadow và phần chưa thử
    trên hardware; không gọi bản hiện tại là production/competition-ready.

## 15. Prompt gợi ý để tiếp tục bằng Antigravity

```text
Hãy đọc src/smart_city/v2/README.md và docs/SMART_CITY_V2_GUIDE.md trước.
Tiếp quản Smart City V2, không sửa các thay đổi ngoài src/smart_city/v2,
src/smart_city/main_smart_city_v2.py, tests/test_smart_city_v2_*.py và tài liệu
liên quan. Đầu tiên chạy full unittest bằng discover, Python 3.6 grammar và
offline image smoke. Kiểm tra kỹ patch cuối về actuator E-stop latch, semantic
sequence/source TTL, requires_sign và planned-side keep-out; viết regression
trước mọi sửa đổi.
Không bật motor, không đổi calibrated/validated_for_live, và không dùng route
mẫu làm route thi. Sau đó báo các blocker còn lại và đề xuất bước calibration
dựa trên video camera CSI thật.
```
