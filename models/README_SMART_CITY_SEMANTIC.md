# Smart City semantic checkpoint

`smart_city_semantic_best.pt` is the YOLOv8n detect checkpoint extracted from
the team hand-off archive `finally.zip` (`weights/best.pt`). It is deliberately
named separately from the older lane/crosswalk `models/best.pt` used on other
branches.

- Expected SHA-256:
  `e1cfa66871e03ab79f4a0d4bfeb1417bc57614f6c3079894b68a4594f376db3a`
- Task: object detection (not segmentation)
- Classes: `Corner`, `Decision`, `Forbidden`, `Green_Light`, `Interact`,
  `Left`, `Red_Light`, `Right`, `straight`
- Runtime adapter: `src/smart_city/v2/yolo_semantic.py`

Only sign/light classes enter the semantic FSM. `Corner`, `Decision` and
`Interact` are ignored. Do not enable motors until replay and ROS shadow tests
show fresh, stable results within the configured semantic TTL.

## Verification on 2026-08-21

- The checkpoint loaded successfully with Ultralytics 8.4.23.
- One 640 px CPU smoke inference on the development laptop took about 156 ms.
- On all 47 supplied test images, 640 px inference produced no `Green_Light`,
  `Red_Light` or `Left` boxes at confidence 0.10 or above.
- Retesting at 960 px and confidence 0.01 still produced none of those three
  classes. It did produce `Corner`, `Decision`, `Forbidden`, `Interact`,
  `Right` and `straight`.

Therefore this checkpoint is integrated for **shadow evaluation only**. With
`--require-ai`, the absence of a fresh GREEN correctly keeps the FSM stopped.
The light/Left classes need more labelled data and retraining before a live
course run. Do not lower the confidence threshold merely to bypass the hold.
