# Smart City V2 — Bài 2 standalone

Nhánh này chỉ chứa mã nguồn, model, test và tài liệu cần cho Bài 2 Smart City.
Không chứa Speed Track, archive các bài khác hay tài liệu chung của toàn đội.

## Kiến trúc

```text
Camera
├── OpenCV geometry: lane, vạch giao lộ, đảo xanh/cam và biên sân
└── YOLO semantic: biển báo + đèn
        ↓
Scenario + allowed exits
        ↓
FSM quyết định hành động
        ↓
Safety supervisor
        ↓
Steering/throttle
```

AI không được xuất steering, throttle, topology hoặc crosswalk. Model chạy bằng
worker latest-frame-only; kết quả stale/out-of-order bị loại.

## Kiểm tra sau khi checkout

```bash
python3 -B -m unittest discover -s tests -p "test_smart_city_v2_*.py"
sha256sum models/smart_city_semantic_best.pt
python3 -c "import ultralytics; print(ultralytics.__version__)"
```

Hash checkpoint đúng:

```text
f5fd08620edc8b7d5da26a5bd977a1ca17cf2c2a8e8bbc1874efb77af51512f2
```

## Chạy ROS shadow — không motor

```bash
source /opt/ros/melodic/setup.bash
source /home/jetson/catkin_ws/devel/setup.bash

python3 -B src/smart_city/main_smart_city_v2.py \
  --ros \
  --semantic-model models/smart_city_semantic_best.pt \
  --require-ai \
  --scenario src/smart_city/v2/scenario_example.json \
  --config src/smart_city/v2/config_example.json \
  --web-port 8080 \
  --log /home/jetson/smart_city_ai_shadow.csv
```

Không thêm `--enable-motors` ở bước này.

## Trạng thái model và blocker hiện tại

Checkpoint 6 lớp đã được train lại từ dataset sửa lỗi nhãn box/polygon. Trên
validation, recall của `Green_Light`, `Left`, `Red_Light` lần lượt là 1,00;
0,95; 0,60. Trên test độc lập là 1,00; 1,00; 0,50, nhưng test chỉ có hai mẫu
đèn đỏ nên chưa đủ để kết luận khả năng tổng quát ngoài sân thật. Ở ngưỡng live
0,60, model ưu tiên an toàn: đèn quá nhỏ có thể làm xe tiếp tục chờ thay vì đoán
GREEN.

Model vẫn phải qua ROS shadow bằng camera gắn xe trước khi bật motor. Blocker
phần cứng lần gần nhất là actuator trực tiếp không thấy I2C `0x60` và rơi vào
mock; cần nối runner với đúng driver ROS/serial đang điều khiển xe. Xem
`docs/SMART_CITY_V2_GUIDE.md` và `models/README_SMART_CITY_SEMANTIC.md` trước
khi test thật.
