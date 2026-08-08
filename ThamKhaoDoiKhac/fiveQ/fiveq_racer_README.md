# fiveq_racer (ROS 1 Melodic)

Reactive lane-following brain for the Waveshare JetRacer ROS AI Kit. It
subscribes the CSI camera, computes a lane offset, and publishes `/cmd_vel`;
the kit's chassis node turns that into steering + throttle.

## Install
```bash
cp -r fiveq_racer ~/catkin_ws/src/
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Run (4 terminals, source devel/setup.bash in each)
```bash
roslaunch jetracer jetracer.launch         # 1. chassis  (subscribes /cmd_vel)
roslaunch jetracer csi_camera.launch       # 2. camera
roslaunch fiveq_racer lane_follow.launch    # 3. the brain (starts DISABLED)
```
Put the car on the track, then enable driving:
```bash
rostopic pub /fiveq/enable std_msgs/Bool "data: true"    # go
rostopic pub /fiveq/enable std_msgs/Bool "data: false"   # stop
```
Ctrl+C on the brain also sends a zero Twist to stop the car.

## Tuning (no rebuild needed; edit launch params or use rosparam then relaunch)
- `mask_threshold` : lower (~120-150) if bright floor gets picked up as lane.
- `cruise_speed`   : raise once it tracks well.
- `kp` / `kd`      : raise kp if it under-steers in corners; raise kd if it wobbles.

## See what it sees (no monitor needed on the car)
From any machine on the same ROS master:
```bash
rqt_image_view /fiveq/debug/compressed
```

## Notes
- No cv_bridge dependency: JPEG is decoded directly with OpenCV.
- Lane detection is classical CV for now; swap in the U-Net (lane_unet.pth)
  behind the same offset output later.
- Obstacle stop via /scan is optional (set use_lidar:=true and run lidar.launch).
