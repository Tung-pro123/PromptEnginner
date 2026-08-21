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
e1cfa66871e03ab79f4a0d4bfeb1417bc57614f6c3079894b68a4594f376db3a
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

## Blocker hiện tại

Checkpoint load và inference được, nhưng không phát hiện `Green_Light`,
`Red_Light` hoặc `Left` trên 47 ảnh test bàn giao, kể cả raw confidence 0.01.
Do đó model hiện chỉ dùng shadow; `--require-ai` sẽ giữ xe an toàn khi thiếu
GREEN. Ngoài ra actuator trực tiếp hiện chưa khớp phần cứng ROS/serial của xe.
Xem `docs/SMART_CITY_V2_GUIDE.md` và
`models/README_SMART_CITY_SEMANTIC.md` trước khi test thật.
