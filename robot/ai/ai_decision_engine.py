"""
AI Decision Engine - Bộ não AI điều phối hành vi cấp cao
=========================================================
Module này chứa logic AI để đưa ra quyết định hành vi cho robot,
bao gồm: tuân thủ đèn giao thông, rẽ trái/phải/thẳng theo biển báo,
bám làn đường.

Thiết kế theo kiến trúc Rule-Based + Sensor Fusion:
- Input: Dữ liệu từ Blackboard (Camera, Traffic Light/Sign)
- Output: Lệnh hành vi (AI_ACTION) ghi lại vào Blackboard

Thứ tự ưu tiên (Priority Chain):
  1. WAIT_RED_LIGHT  - Đèn đỏ -> dừng chờ
  2. TURN / GO_STRAIGHT - Ngã tư theo biển báo chỉ dẫn
  3. FOLLOW_LANE     - Bám làn thông thường (mặc định)
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
    WAIT_RED_LIGHT = "WAIT_RED_LIGHT"   # Dừng chờ đèn đỏ




# ============================================================
# Bộ não AI chính - AIDecisionEngine
# ============================================================
class AIDecisionEngine:
    """
    Bộ não AI điều phối hành vi cấp cao của Robot.

    Nguyên lý hoạt động (Priority Chain):
      1. WAIT_RED_LIGHT  - Đèn đỏ -> dừng chờ
      2. TURN (rẽ)       - Phát hiện ngã tư + đọc biển báo chỉ dẫn
      3. FOLLOW_LANE     - Bám làn thông thường (mặc định)
    """

    def __init__(self):
        self.current_action = Action.FOLLOW_LANE
        self.action_start_time = time.time()
        self.pending_turn = None   # Hướng rẽ đang chờ (ghi nhớ từ biển báo)

        # Bộ nhớ ngắn: đếm số frame liên tiếp phát hiện ngã tư
        self._intersection_frame_count = 0
        self._intersection_confirm_frames = settings.AI_INTERSECTION_MIN_FRAMES

        # Hướng rẽ mặc định tại ngã tư nếu không có biển báo
        self.turn_priority = list(settings.AI_TURN_PRIORITY)

    # ----------------------------------------------------------
    # API công khai
    # ----------------------------------------------------------
    def set_turn_priority(self, priority_list):
        """
        Cho phép task bên ngoài chỉ định thứ tự ưu tiên rẽ mặc định.
        Ví dụ: set_turn_priority(['right', 'straight'])
        """
        self.turn_priority = priority_list

    # ----------------------------------------------------------
    # Logic nội bộ
    # ----------------------------------------------------------
    def _set_action(self, action):
        if self.current_action != action:
            self.current_action = action
            self.action_start_time = time.time()

    def _action_elapsed(self):
        return time.time() - self.action_start_time

    def _detect_intersection(self, waypoint_count):
        """
        Phát hiện ngã tư dựa trên điều kiện:
        - Camera mất vạch hoàn toàn (waypoint_count == 0)
        Cần N frame liên tiếp để xác nhận (tránh nhiễu).
        """
        if waypoint_count == 0:
            self._intersection_frame_count += 1
        else:
            self._intersection_frame_count = 0

        return self._intersection_frame_count >= self._intersection_confirm_frames

    def _decide_turn_direction(self):
        """Quyết định hướng rẽ mặc định theo thứ tự ưu tiên đã cài đặt."""
        priority = self.turn_priority[0] if self.turn_priority else 'straight'
        if priority == 'left':
            return Action.TURN_LEFT
        elif priority == 'right':
            return Action.TURN_RIGHT
        else:
            return Action.GO_STRAIGHT

    def compute(self, blackboard):
        """
        Hàm cốt lõi: đọc Blackboard -> tính toán -> trả về Action + params điều khiển.

        Returns:
            action (str): Hành vi hiện tại (từ class Action)
            steer (float): Góc lái (-1.0 đến 1.0)
            throttle (float): Ga (0 = dừng, dương = tiến)
        """
        # --- Đọc dữ liệu từ Blackboard ---
        lane_waypoints = blackboard.get('lane_waypoints', [])
        steering_pid   = blackboard.get('steering', 0.0)   # Góc lái đề xuất từ Controller
        traffic_light  = blackboard.get('traffic_light', 'NONE')
        traffic_sign   = blackboard.get('traffic_sign', 'NONE')

        waypoint_count = len(lane_waypoints)

        # ===================================================
        # PRIORITY 1: ĐÈN ĐỎ -> DỪNG CHỜ
        # ===================================================
        if traffic_light == 'RED':
            self._set_action(Action.WAIT_RED_LIGHT)
            import rospy
            rospy.logdebug("[AI] ĐÈN ĐỎ! Dừng xe chờ...")
            return Action.WAIT_RED_LIGHT, 0.0, 0.0

        # Thoát khỏi WAIT_RED_LIGHT khi đèn chuyển xanh (hoặc NONE)
        if self.current_action == Action.WAIT_RED_LIGHT and traffic_light != 'RED':
            import rospy
            rospy.logdebug("[AI] ĐÈN XANH! Tiếp tục hành trình.")
            self._set_action(Action.FOLLOW_LANE)

        # ===================================================
        # PRIORITY 2: GHI NHỚ BIỂN BÁO CHỈ DẪN
        # ===================================================
        # Biển báo được ghi nhớ để dùng khi đến ngã tư
        if traffic_sign == 'LEFT' and self.current_action != Action.TURN_LEFT:
            self.pending_turn = 'left'
        elif traffic_sign == 'RIGHT' and self.current_action != Action.TURN_RIGHT:
            self.pending_turn = 'right'
        elif traffic_sign == 'STRAIGHT' and self.current_action != Action.GO_STRAIGHT:
            self.pending_turn = 'straight'

        # ===================================================
        # PRIORITY 3: RẼ TẠI NGÃ TƯ (phát hiện mất vạch)
        # ===================================================
        if self._detect_intersection(waypoint_count):
            if self.current_action not in (Action.TURN_LEFT, Action.TURN_RIGHT, Action.GO_STRAIGHT):
                # Ưu tiên biển báo nếu có, không thì dùng turn_priority mặc định
                if self.pending_turn == 'left':
                    self._set_action(Action.TURN_LEFT)
                elif self.pending_turn == 'right':
                    self._set_action(Action.TURN_RIGHT)
                elif self.pending_turn == 'straight':
                    self._set_action(Action.GO_STRAIGHT)
                else:
                    self._set_action(self._decide_turn_direction())

                self._intersection_frame_count = 0  # Reset bộ đếm
                self.pending_turn = None             # Xóa biển sau khi đã dùng

                import rospy
                rospy.logdebug(f"[AI] NGÃ TƯ! -> {self.current_action}")

        if self.current_action == Action.TURN_LEFT:
            if self._action_elapsed() < settings.AI_TURN_HOLD_TIME:
                import rospy
                rospy.logdebug(f"[AI] TURN_LEFT | Còn {settings.AI_TURN_HOLD_TIME - self._action_elapsed():.1f}s")
                return Action.TURN_LEFT, settings.AI_TURN_LEFT_STEER, settings.AI_TURN_THROTTLE
            else:
                self._set_action(Action.FOLLOW_LANE)

        if self.current_action == Action.TURN_RIGHT:
            if self._action_elapsed() < settings.AI_TURN_HOLD_TIME:
                import rospy
                rospy.logdebug(f"[AI] TURN_RIGHT | Còn {settings.AI_TURN_HOLD_TIME - self._action_elapsed():.1f}s")
                return Action.TURN_RIGHT, settings.AI_TURN_RIGHT_STEER, settings.AI_TURN_THROTTLE
            else:
                self._set_action(Action.FOLLOW_LANE)

        if self.current_action == Action.GO_STRAIGHT:
            if self._action_elapsed() < settings.AI_TURN_HOLD_TIME:
                import rospy
                rospy.logdebug(f"[AI] GO_STRAIGHT (theo biển) | Còn {settings.AI_TURN_HOLD_TIME - self._action_elapsed():.1f}s")
                return Action.GO_STRAIGHT, 0.0, settings.AI_TURN_THROTTLE
            else:
                self._set_action(Action.FOLLOW_LANE)

        # ===================================================
        # DEFAULT: FOLLOW_LANE (Bám làn thông thường)
        # ===================================================
        self._set_action(Action.FOLLOW_LANE)
        return Action.FOLLOW_LANE, steering_pid, settings.AI_SPEED_NORMAL

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
