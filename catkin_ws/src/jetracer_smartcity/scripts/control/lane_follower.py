import cv2
import numpy as np

class LaneFollower:
    def __init__(self, kp=0.005, kd=0.002, ki=0.0):
        # PID coefficients
        self.kp = kp
        self.kd = kd
        self.ki = ki
        
        self.prev_error = 0.0
        self.integral = 0.0
        
        # Lane threshold parameters
        self.min_intensity = 180  # Threshold for white lane lines
        
        # ROI Crop parameters (relative to frame dimensions)
        self.roi_ymin = 0.6
        self.roi_ymax = 0.9

    def get_lane_mask(self, frame):
        """
        Creates a binary mask of the lane line within the ROI.
        """
        h, w = frame.shape[:2]
        
        # Crop region of interest
        ymin = int(h * self.roi_ymin)
        ymax = int(h * self.roi_ymax)
        roi = frame[ymin:ymax, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to isolate white dashed lines
        _, thresh = cv2.threshold(blurred, self.min_intensity, 255, cv2.THRESH_BINARY)
        
        return thresh, ymin

    def compute_steering(self, frame):
        """
        Processes image frame, computes lane error, runs PID, and returns steering command [-1.0, 1.0].
        """
        if frame is None or frame.size == 0:
            return 0.0
            
        h, w = frame.shape[:2]
        center_x = w / 2.0
        
        # Get mask
        mask, ymin = self.get_lane_mask(frame)
        
        # Find contours of the white lines
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: if no line is detected, decay error to zero but maintain heading
            error = self.prev_error * 0.9
        else:
            # Find the largest contour (representing the primary center dashed line)
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                # Calculate centroid x-coordinate
                centroid_x = int(M["m10"] / M["m00"])
                error = centroid_x - center_x
            else:
                error = self.prev_error
                
        # PID Controller calculation
        dt = 0.033  # ~30 FPS
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        steering = self.kp * error + self.kd * derivative + self.ki * self.integral
        self.prev_error = error
        
        # Clip steering output to vehicle limits [-1.0, 1.0]
        steering = np.clip(steering, -1.0, 1.0)
        
        return float(steering)

if __name__ == '__main__':
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32
    # CvBridge is used to convert ROS Image message to OpenCV BGR frame
    # Fallback to manual numpy conversion if CvBridge is missing in a raw Python 3 environment
    try:
        from cv_bridge import CvBridge
        bridge = CvBridge()
        def get_cv_image(msg):
            return bridge.imgmsg_to_cv2(msg, "bgr8")
    except ImportError:
        def get_cv_image(msg):
            # Manual fallback conversion for standard height/width
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    rospy.init_node('lane_follower', anonymous=True)
    
    # PID parameters can be adjusted via ROS parameters
    kp = rospy.get_param('~kp', 0.005)
    kd = rospy.get_param('~kd', 0.002)
    ki = rospy.get_param('~ki', 0.0)
    
    lf = LaneFollower(kp=kp, kd=kd, ki=ki)
    
    steering_pub = rospy.Publisher('/steering_angle', Float32, queue_size=10)
    
    def image_callback(msg):
        try:
            cv_img = get_cv_image(msg)
            steering_angle = lf.compute_steering(cv_img)
            steering_pub.publish(Float32(steering_angle))
        except Exception as e:
            rospy.logwarn("Lane follower frame processing failed: %s", str(e))
            
    rospy.Subscriber('/image_raw', Image, image_callback, queue_size=1)
    rospy.loginfo("Lane Follower ROS Node initialized.")
    rospy.spin()

