#!/usr/bin/env python3
"""FiveQ reactive lane-following node for the Waveshare JetRacer ROS AI Kit.

Pipeline (same brain validated in the DonkeyCar sim, now as a rospy node):
    /csi_cam_0/image_raw/compressed  (JPEG in)
        -> decode -> lane mask -> lateral offset -> PID -> Twist
    -> /cmd_vel  (the chassis node turns this into servo + motor commands)

Self-contained on purpose: no cv_bridge (we decode JPEG with cv2 directly),
no extra packages. Runs under python3 rospy.

SAFETY: the node starts DISABLED and only drives after you enable it:
    rostopic pub /fiveq/enable std_msgs/Bool "data: true"   # go
    rostopic pub /fiveq/enable std_msgs/Bool "data: false"  # stop
Ctrl+C also sends a final zero Twist so the car stops.

Prereqs (start these first, each in its own terminal):
    roslaunch jetracer jetracer.launch        # chassis  -> subscribes /cmd_vel
    roslaunch jetracer csi_camera.launch      # camera   -> publishes the image
    roslaunch jetracer lidar.launch           # optional -> /scan (obstacle stop)
"""
import math

import numpy as np
import cv2

import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist


# ----------------------------- vision helpers -----------------------------
def region_of_interest(img, top_ratio):
    h = img.shape[0]
    y0 = int(h * top_ratio)
    out = np.zeros_like(img)
    out[y0:, ...] = img[y0:, ...]
    return out


def lane_mask(bgr, thresh):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return mask


def lane_offset_from_mask(mask):
    """Return (offset in [-1, 1], valid). Positive = lane center to the right."""
    h, w = mask.shape[:2]
    band = mask[int(h * 0.7):, :]
    xs = np.where(band > 0)[1]
    if xs.size < 50:
        return 0.0, False
    centroid = float(xs.mean())
    offset = (centroid - w / 2.0) / (w / 2.0)
    return float(np.clip(offset, -1.0, 1.0)), True


class SteerPID(object):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev = 0.0

    def step(self, offset, dt):
        if dt <= 0:
            dt = 1e-3
        deriv = (offset - self.prev) / dt
        self.prev = offset
        return self.kp * offset + self.kd * deriv

    def reset(self):
        self.prev = 0.0


# ----------------------------- the node -----------------------------
class LaneFollowNode(object):
    def __init__(self):
        rospy.init_node("fiveq_lane_follow")

        # topics
        self.image_topic = rospy.get_param("~image_topic",
                                           "/csi_cam_0/image_raw/compressed")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

        # driving params (tune live with rosparam or rqt_reconfigure-free set)
        self.cruise = rospy.get_param("~cruise_speed", 0.30)   # linear.x
        self.kp = rospy.get_param("~kp", 0.9)
        self.kd = rospy.get_param("~kd", 0.25)
        self.max_yaw = rospy.get_param("~max_yaw", 1.0)        # rad/s clamp
        self.threshold = rospy.get_param("~mask_threshold", 180)
        self.roi_top = rospy.get_param("~roi_top", 0.55)

        # lidar safety (off unless lidar.launch is running)
        self.use_lidar = rospy.get_param("~use_lidar", False)
        self.stop_distance = rospy.get_param("~stop_distance", 0.35)
        self.front_arc = math.radians(rospy.get_param("~front_arc_deg", 40.0))

        # debug overlay (view with rqt_image_view /fiveq/debug/compressed)
        self.publish_debug = rospy.get_param("~publish_debug", True)

        self.pid = SteerPID(self.kp, self.kd)
        self.enabled = False
        self.blocked = False
        self.last_time = rospy.Time.now()

        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        if self.publish_debug:
            self.pub_dbg = rospy.Publisher("/fiveq/debug/compressed",
                                           CompressedImage, queue_size=1)

        rospy.Subscriber("/fiveq/enable", Bool, self.on_enable, queue_size=1)
        rospy.Subscriber(self.image_topic, CompressedImage, self.on_image,
                         queue_size=1, buff_size=2 ** 22)
        if self.use_lidar:
            rospy.Subscriber("/scan", LaserScan, self.on_scan, queue_size=1)

        rospy.on_shutdown(self.stop_car)
        rospy.loginfo("fiveq_lane_follow ready (DISABLED). "
                      "Enable with: rostopic pub /fiveq/enable std_msgs/Bool \"data: true\"")

    # ---- callbacks ----
    def on_enable(self, msg):
        self.enabled = msg.data
        rospy.loginfo("fiveq_lane_follow %s", "ENABLED" if self.enabled else "DISABLED")
        if not self.enabled:
            self.stop_car()
            self.pid.reset()

    def on_scan(self, scan):
        blocked = False
        angle = scan.angle_min
        for r in scan.ranges:
            if -self.front_arc / 2 <= angle <= self.front_arc / 2:
                if math.isfinite(r) and 0.0 < r < self.stop_distance:
                    blocked = True
                    break
            angle += scan.angle_increment
        self.blocked = blocked

    def on_image(self, msg):
        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        self.last_time = now

        arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        mask = lane_mask(frame, int(self.threshold))
        mask = region_of_interest(mask, self.roi_top)
        offset, valid = lane_offset_from_mask(mask)

        cmd = Twist()
        if self.enabled and not self.blocked:
            if valid:
                yaw = -(self.pid.step(offset, dt))   # +offset (lane right) -> steer right
                yaw = max(-self.max_yaw, min(self.max_yaw, yaw))
                cmd.linear.x = float(self.cruise)
                cmd.angular.z = float(yaw)
            else:
                self.pid.reset()
                cmd.linear.x = float(self.cruise * 0.4)   # creep straight
                cmd.angular.z = 0.0
        self.pub_cmd.publish(cmd)

        rospy.loginfo_throttle(
            0.5, "offset=%+.3f valid=%s blocked=%s -> lin=%.2f ang=%+.2f (en=%s)"
            % (offset, valid, self.blocked, cmd.linear.x, cmd.angular.z, self.enabled))

        if self.publish_debug:
            self.publish_overlay(frame, mask, offset, valid, cmd)

    # ---- helpers ----
    def publish_overlay(self, frame, mask, offset, valid, cmd):
        vis = frame.copy()
        h, w = vis.shape[:2]
        tint = np.zeros_like(vis)
        tint[mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 1.0, tint, 0.35, 0)
        cx = w // 2
        cv2.line(vis, (cx, h - 1), (cx, int(h * 0.6)), (200, 200, 200), 1)
        if valid:
            mxp = int(cx + offset * (w / 2))
            cv2.line(vis, (mxp, h - 1), (mxp, int(h * 0.72)), (0, 165, 255), 3)
        txt = "off %+.2f  lin %.2f  ang %+.2f  en=%s" % (
            offset, cmd.linear.x, cmd.angular.z, self.enabled)
        cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        ok, jpg = cv2.imencode(".jpg", vis)
        if ok:
            out = CompressedImage()
            out.header.stamp = rospy.Time.now()
            out.format = "jpeg"
            out.data = jpg.tobytes()
            self.pub_dbg.publish(out)

    def stop_car(self):
        try:
            self.pub_cmd.publish(Twist())
        except Exception:
            pass


if __name__ == "__main__":
    try:
        LaneFollowNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
