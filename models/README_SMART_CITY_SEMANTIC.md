# Smart City semantic checkpoint

`smart_city_semantic_best.pt` is the corrected YOLOv8n detect checkpoint for
Smart City signs and traffic lights. It is deliberately named separately from
older lane/crosswalk checkpoints used on other branches.

- Expected SHA-256:
  `f5fd08620edc8b7d5da26a5bd977a1ca17cf2c2a8e8bbc1874efb77af51512f2`
- Task: object detection (not segmentation)
- Classes: `Forbidden`, `Green_Light`, `Left`, `Red_Light`, `Right`, `straight`
- Runtime adapter: `src/smart_city/v2/yolo_semantic.py`

Scene-geometry classes were removed because geometry belongs to OpenCV and the
deterministic controller. Do not enable motors until replay and ROS shadow
tests show fresh, stable results within the configured semantic TTL.

## Retraining fix and verification on 2026-08-21

The source Roboflow export mixed five-column boxes and segmentation polygons
inside 361 label files. The preparation script converts semantic polygons to
tight boxes, removes `Corner`/`Decision`/`Interact`, adds upper-image light
crops and balances the rare `Green_Light`, `Left` and `Red_Light` classes.

Validation at 640 px (92 images, 105 instances):

| Class | Instances | Precision | Recall | mAP50 |
|---|---:|---:|---:|---:|
| Forbidden | 30 | 0.992 | 0.933 | 0.985 |
| Green_Light | 6 | 0.840 | 1.000 | 0.948 |
| Left | 13 | 1.000 | 0.950 | 0.995 |
| Red_Light | 10 | 0.926 | 0.600 | 0.767 |
| Right | 28 | 0.965 | 0.980 | 0.984 |
| straight | 18 | 0.984 | 0.944 | 0.948 |

Independent test at 640 px (47 images, 57 instances) gave overall precision
0.828, recall 0.905 and mAP50 0.945. `Green_Light` recall was 1.00 (8 samples),
`Left` 1.00 (7), and `Red_Light` 0.50 (only 2). Because the red sample is tiny
and the official camera/course domain can differ, this is **shadow-ready, not
proof of live-course readiness**. Keep live confidence at 0.60 until real
camera replay shows a justified change; missing GREEN must hold the car.

Rebuild and train (the dataset ZIP itself is intentionally not committed):

```bash
python3 -B tools/prepare_smart_city_semantic_dataset.py \
  "Smart-city version2 new idea.v1i.yolov8.zip" \
  training/semantic_dataset_light_roi \
  --oversample-min-count 144 --light-crop-bottom 0.58

python3 -B src/smart_city/train_semantic_yolo.py \
  --data training/semantic_dataset_light_roi/data.yaml \
  --model models/smart_city_semantic_best.pt \
  --epochs 20 --imgsz 640 --batch 8 --device cpu
```
