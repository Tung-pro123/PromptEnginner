"""
AI Decision Engine - Bộ não AI điều phối hành vi cấp cao
=========================================================
Module này chứa logic AI để đưa ra quyết định hành vi phức tạp cho robot,
bao gồm: rẽ trái, rẽ phải, đi thẳng, dừng, đợi chờ tín hiệu, ...

Thiết kế theo kiến trúc Rule-Based + Sensor Fusion:
- Input: Dữ liệu từ Blackboard (Lidar, Camera, FSM State, Traffic Light/Sign)
- Output: Lệnh hành vi (AI_ACTION) ghi lại vào Blackboard

Hành vi (Action) được định nghĩa rõ ràng để controller cấp dưới thực thi.
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


# ============================================================
# Định nghĩa tất cả hành vi mà AI có thể ra lệnh
# ============================================================
class Action:
    FOLLOW_LANE    = "FOLLOW_LANE"      # Bám vạch, đi thẳng theo làn
    TURN_LEFT      = "TURN_LEFT"        # Rẽ trái (tại ngã tư / giao lộ)
    TURN_RIGHT     = "TURN_RIGHT"       # Rẽ phải (tại ngã tư / giao lộ)
    GO_STRAIGHT    = "GO_STRAIGHT"      # Đi thẳng qua giao lộ (theo biển)
    WAIT           = "WAIT"             # Đợi (ví dụ đèn đỏ, chướng ngại vật cứng)
    WAIT_RED_LIGHT = "WAIT_RED_LIGHT"   # Dừng chờ đèn đỏ (riêng biệt)
    EMERGENCY_STOP = "EMERGENCY_STOP"   # Dừng khẩn cấp (vật cản cực gần)
    REVERSE        = "REVERSE"          # Lùi xe (khi bị kẹt)
    DODGE_LEFT     = "DODGE_LEFT"       # Né tránh sang trái
    DODGE_RIGHT    = "DODGE_RIGHT"      # Né tránh sang phải


# ============================================================
# Các ngưỡng riêng của AI (độc lập với settings)
# ============================================================
class AIConfig:
    # Ngưỡng khoảng cách Lidar để ra quyết định
    EMERGENCY_STOP_DIST   = 0.20   # m - Dừng khẩn cấp
    HARD_BRAKE_DIST       = 0.40   # m - Giảm tốc mạnh
    WAIT_DIST             = 0.55   # m - Đứng yên chờ
    CAUTION_DIST          = 0.90   # m - Cảnh giác, chạy chậm lại

    # Ngưỡng phát hiện ngã tư / giao lộ (dựa trên số vạch trắng)
    INTERSECTION_WAYPOINT_COUNT = 0   # Nếu camera không tìm được waypoint = mất vạch = có thể là ngã tư

    # Thời gian tối thiểu giữ nguyên hành vi (tránh rung lắc)
    MIN_ACTION_HOLD_TIME  = 0.8    # giây
    TURN_HOLD_TIME        = 2.5    # giây - giữ nguyên lệnh rẽ
    WAIT_TIMEOUT          = 5.0    # giây - đợi tối đa rồi tiến

    # Góc lái khi rẽ cứng
    TURN_LEFT_STEER       = -0.85
    TURN_RIGHT_STEER      = +0.85
    TURN_THROTTLE         = 0.18   # Tốc độ chậm khi rẽ

    # Tốc độ theo trạng thái
    SPEED_NORMAL          = 0.22
    SPEED_CAUTION         = 0.14
    SPEED_REVERSE         = -0.15


# ============================================================
# Bộ não AI chính - AIDecisionEngine
# ============================================================
class AIDecisionEngine:
    """
    Bộ não AI điều phối hành vi cấp cao của Robot.

    Nguyên lý hoạt động (Priority Chain):
      1. EMERGENCY_STOP  - Vật cản cực gần -> dừng ngay
      2. WAIT_RED_LIGHT  - Đèn đỏ -> dừng chờ
      3. WAIT            - Vật cản gần, không thể đi -> đứng chờ
      4. TURN (rẽ)       - Phát hiện ngã tư/giao lộ hoặc biển chỉ dẫn
      5. DODGE           - Né tránh vật cản động
      6. FOLLOW_LANE     - Bám làn thông thường (mặc định)
    """

    def __init__(self):
        self.current_action = Action.FOLLOW_LANE
        self.action_start_time = time.time()
        self.pending_turn = None          # Hướng rẽ đang chờ thực hiện
        self.wait_start_time = 0.0
        self.reverse_start_time = 0.0
        self.is_reversing = False

        # Bộ nhớ ngắn: đếm số frame liên tiếp phát hiện ngã tư
        self._intersection_frame_count = 0
        self._intersection_confirm_frames = 5  # Cần 5 frame liên tiếp

        # Hướng rẽ mặc định tại ngã tư (có thể override từ ngoài)
        # Thứ tự ưu tiên: 'left', 'right', 'straight'
        self.turn_priority = ['left', 'right', 'straight']

    # ----------------------------------------------------------
    # API công khai để điều chỉnh hướng rẽ ưu tiên từ bên ngoài
    # ----------------------------------------------------------
    def set_turn_priority(self, priority_list):
        """
        Cho phép task bên ngoài chỉ định thứ tự ưu tiên rẽ.
        Ví dụ: set_turn_priority(['right', 'straight'])
        """
        self.turn_priority = priority_list

    def command_turn(self, direction):
        """Chủ động ra lệnh rẽ ngay lập tức (từ logic bên ngoài)."""
        if direction == 'left':
            self._set_action(Action.TURN_LEFT)
        elif direction == 'right':
            self._set_action(Action.TURN_RIGHT)

    # ----------------------------------------------------------
    # Logic nội bộ
    # ----------------------------------------------------------
    def _set_action(self, action):
        if self.current_action != action:
            self.current_action = action
            self.action_start_time = time.time()

    def _action_elapsed(self):
        return time.time() - self.action_start_time

    def _detect_intersection(self, waypoint_count, front_dist):
        """
        Phát hiện ngã tư dựa trên 2 điều kiện:
        1. Camera mất vạch (waypoint_count == 0) - đang đi vào giao lộ
        2. Lidar không phát hiện vật cản gần phía trước
        """
        is_intersection_condition = (
            waypoint_count == AIConfig.INTERSECTION_WAYPOINT_COUNT
            and front_dist > AIConfig.CAUTION_DIST
        )
        if is_intersection_condition:
            self._intersection_frame_count += 1
        else:
            self._intersection_frame_count = 0

        return self._intersection_frame_count >= self._intersection_confirm_frames

    def _decide_turn_direction(self):
        """Quyết định hướng rẽ theo thứ tự ưu tiên đã cài đặt."""
        priority = self.turn_priority[0] if self.turn_priority else 'straight'
        if priority == 'left':
            return Action.TURN_LEFT
        elif priority == 'right':
            return Action.TURN_RIGHT
        else:
            return Action.FOLLOW_LANE  # Đi thẳng qua giao lộ

    def compute(self, blackboard):
        """
        Hàm cốt lõi: đọc Blackboard -> tính toán -> trả về Action + params điều khiển.

        Returns:
            action (str): Hành vi hiện tại (từ class Action)
            steer (float): Góc lái (-1.0 đến 1.0)
            throttle (float): Ga (âm = lùi, 0 = dừng, dương = tiến)
        """
        # --- Đọc dữ liệu từ Blackboard ---
        front_dist      = blackboard.get('front_dist', 999.0)
        closest_angle   = blackboard.get('closest_angle', 0.0)
        side_clear      = blackboard.get('side_clear', True)
        lane_waypoints  = blackboard.get('lane_waypoints', [])
        center_x        = blackboard.get('center_x', 150.0)
        steering_pid    = blackboard.get('steering', 0.0)  # Kết quả từ controller PID/Predictive
        traffic_light   = blackboard.get('traffic_light', 'NONE')
        traffic_sign    = blackboard.get('traffic_sign', 'NONE')

        waypoint_count = len(lane_waypoints)

        # ===================================================
        # PRIORITY 1: EMERGENCY STOP
        # ===================================================
        if front_dist < AIConfig.EMERGENCY_STOP_DIST:
            self._set_action(Action.EMERGENCY_STOP)
            import rospy
            rospy.logdebug(f"[AI] EMERGENCY STOP! Vật cản chỉ cách {front_dist:.2f}m!")
            return Action.EMERGENCY_STOP, 0.0, 0.0

        # Thoát khỏi EMERGENCY_STOP sau khi vật cản rời đi
        if self.current_action == Action.EMERGENCY_STOP and front_dist > AIConfig.WAIT_DIST:
            self._set_action(Action.FOLLOW_LANE)

        # ===================================================
        # PRIORITY 2: ĐÈN ĐỎ -> DẪNg CHỜN
        # ===================================================
        if traffic_light == 'RED':
            if self.current_action != Action.WAIT_RED_LIGHT:
                self._set_action(Action.WAIT_RED_LIGHT)
            import rospy
            rospy.logdebug("[AI] ĐÈN ĐỎ! Dừng xe chờ...")
            return Action.WAIT_RED_LIGHT, 0.0, 0.0

        # Thoát khỏi WAIT_RED_LIGHT khi đèn chuyển xanh
        if self.current_action == Action.WAIT_RED_LIGHT and traffic_light != 'RED':
            import rospy
            rospy.logdebug("[AI] ĐÈN XANH! Tiếp tục hành trình.")
            self._set_action(Action.FOLLOW_LANE)

        # ===================================================
        # PRIORITY 3: WAIT (Vật cản gần, không có lối thoát)
        # ===================================================
        if front_dist < AIConfig.WAIT_DIST and not side_clear:
            if self.current_action != Action.WAIT:
                self.wait_start_time = time.time()
                self._set_action(Action.WAIT)
            # Nếu đợi quá lâu -> thử lùi xe
            if time.time() - self.wait_start_time > AIConfig.WAIT_TIMEOUT:
                self._set_action(Action.REVERSE)
                self.reverse_start_time = time.time()
            import rospy
            rospy.logdebug(f"[AI] WAIT: Vật cản {front_dist:.2f}m, sườn xe bị chặn.")
            return Action.WAIT, 0.0, 0.0

        # ===================================================
        # PRIORITY 3b: REVERSE (Lùi khi bị kẹt sau khi WAIT)
        # ===================================================
        if self.current_action == Action.REVERSE:
            if time.time() - self.reverse_start_time < 1.5:
                import rospy
                rospy.logdebug("[AI] REVERSE: Đang lùi xe thoát khỏi vị trí kẹt...")
                return Action.REVERSE, 0.0, AIConfig.SPEED_REVERSE
            else:
                self._set_action(Action.FOLLOW_LANE)

        # ===================================================
        # PRIORITY 4: DODGE (Né tránh vật cản có lối thoát)
        # ===================================================
        if AIConfig.WAIT_DIST <= front_dist < AIConfig.CAUTION_DIST and side_clear:
            if closest_angle < 0:  # Vật cản bên phải -> né trái
                self._set_action(Action.DODGE_LEFT)
            else:                   # Vật cản bên trái -> né phải
                self._set_action(Action.DODGE_RIGHT)

        if self.current_action in (Action.DODGE_LEFT, Action.DODGE_RIGHT):
            # Sau khi đã né ra khỏi vùng nguy hiểm thì quay lại bình thường
            if front_dist >= AIConfig.CAUTION_DIST or self._action_elapsed() > 3.0:
                self._set_action(Action.FOLLOW_LANE)
            else:
                dodge_steer = -0.6 if self.current_action == Action.DODGE_LEFT else 0.6
                import rospy
                rospy.logdebug(f"[AI] {self.current_action}: steer={dodge_steer:.2f}, dist={front_dist:.2f}m")
                return self.current_action, dodge_steer, AIConfig.SPEED_CAUTION

        # ===================================================
        # PRIORITY 5: BIỂN BÁO CHỈ DẪN (Override hướng rẽ)
        # ===================================================
        # Biển báo có độ ưu tiên cao hơn tự suy luận ngã tư
        if traffic_sign == 'LEFT' and self.current_action not in (Action.TURN_LEFT,):
            self.pending_turn = 'left'
        elif traffic_sign == 'RIGHT' and self.current_action not in (Action.TURN_RIGHT,):
            self.pending_turn = 'right'
        elif traffic_sign == 'STRAIGHT':
            self.pending_turn = 'straight'

        # ===================================================
        # PRIORITY 6: TURN (Rẽ tại ngã tư - phát hiện mất vạch)
        # ===================================================
        if self._detect_intersection(waypoint_count, front_dist):
            if self.current_action not in (Action.TURN_LEFT, Action.TURN_RIGHT, Action.GO_STRAIGHT):
                # Ʈu tiên biển báo nếu có, không thì dùng turn_priority mặc định
                if self.pending_turn == 'left':
                    self._set_action(Action.TURN_LEFT)
                elif self.pending_turn == 'right':
                    self._set_action(Action.TURN_RIGHT)
                elif self.pending_turn == 'straight':
                    self._set_action(Action.GO_STRAIGHT)
                else:
                    turn_action = self._decide_turn_direction()
                    self._set_action(turn_action)
                self._intersection_frame_count = 0  # Reset bộ đếm
                self.pending_turn = None  # Xóa biển báo sau khi đã dùng

        if self.current_action == Action.TURN_LEFT:
            if self._action_elapsed() < AIConfig.TURN_HOLD_TIME:
                import rospy
                rospy.logdebug(f"[AI] TURN_LEFT | Con {AIConfig.TURN_HOLD_TIME - self._action_elapsed():.1f}s")
                return Action.TURN_LEFT, AIConfig.TURN_LEFT_STEER, AIConfig.TURN_THROTTLE
            else:
                self._set_action(Action.FOLLOW_LANE)

        if self.current_action == Action.TURN_RIGHT:
            if self._action_elapsed() < AIConfig.TURN_HOLD_TIME:
                import rospy
                rospy.logdebug(f"[AI] TURN_RIGHT | Con {AIConfig.TURN_HOLD_TIME - self._action_elapsed():.1f}s")
                return Action.TURN_RIGHT, AIConfig.TURN_RIGHT_STEER, AIConfig.TURN_THROTTLE
            else:
                self._set_action(Action.FOLLOW_LANE)

        if self.current_action == Action.GO_STRAIGHT:
            if self._action_elapsed() < AIConfig.TURN_HOLD_TIME:
                import rospy
                rospy.logdebug(f"[AI] GO_STRAIGHT (theo biển) | Con {AIConfig.TURN_HOLD_TIME - self._action_elapsed():.1f}s")
                return Action.GO_STRAIGHT, 0.0, AIConfig.TURN_THROTTLE
            else:
                self._set_action(Action.FOLLOW_LANE)

        # ===================================================
        # DEFAULT: FOLLOW_LANE (Bám làn thông thường)
        # ===================================================
        self._set_action(Action.FOLLOW_LANE)

        # Điều chỉnh tốc độ theo khoảng cách phía trước
        if front_dist < AIConfig.CAUTION_DIST:
            throttle = AIConfig.SPEED_CAUTION
        else:
            throttle = AIConfig.SPEED_NORMAL

        return Action.FOLLOW_LANE, steering_pid, throttle

    def process(self, blackboard):
        """
        Hàm giao tiếp chuẩn với kiến trúc Blackboard.
        Gọi compute() rồi ghi kết quả vào Blackboard để controller thực thi.
        """
        action, steer, throttle = self.compute(blackboard)

        blackboard.set('ai_action', action)
        blackboard.set('ai_steering', steer)
        blackboard.set('ai_throttle', throttle)

        import rospy
        rospy.logdebug(
            f"[AI Engine] Action={action:15s} | Steer={steer:+.3f} | Throttle={throttle:+.2f}")

