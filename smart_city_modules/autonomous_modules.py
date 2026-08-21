import time
import math

class GoStraightModule:
    """
    Module xử lý xe đi thẳng dựa vào tọa độ từ YOLO-Segmentation.
    Lấy tọa độ x_center, y_center so với tâm dưới cùng của ảnh.
    """
    def __init__(self, img_width=640, img_height=480, base_speed=0.3):
        self.img_width = img_width
        self.img_height = img_height
        self.base_speed = base_speed

        # Tọa độ neo (anchor) - tâm dưới cùng của ảnh
        self.anchor_x = self.img_width / 2
        self.anchor_y = self.img_height

    def update_resolution(self, width, height):
        self.img_width = width
        self.img_height = height
        self.anchor_x = self.img_width / 2
        self.anchor_y = self.img_height

    def calculate_command(self, detections):
        """
        Xử lý list detections trả về từ YOLO và xuất ra lệnh lái.
        Format detection mong đợi:
        [
            {"label": "decision", "x": 320, "y": 400},
            {"label": "corner", "x": 100, "y": 200},
            ...
        ]
        """
        corners = [d for d in detections if d["label"] == "corner"]
        decisions = [d for d in detections if d["label"] == "decision"]
        
        target_node = None

        # Quy tắc: Nếu có cả hai, nếu corner xa hơn decision (y_corner < y_decision) 
        # -> ưu tiên lấy decision (vì decision đang gần xe hơn, hệ tọa độ ảnh y=0 ở trên cùng)
        if len(decisions) > 0 and len(corners) > 0:
            closest_decision = max(decisions, key=lambda d: d["y"])
            closest_corner = max(corners, key=lambda c: c["y"])
            
            if closest_corner["y"] < closest_decision["y"]:
                target_node = closest_decision
            else:
                target_node = closest_corner
                
        elif len(decisions) > 0:
            target_node = max(decisions, key=lambda d: d["y"])
        elif len(corners) > 0:
            target_node = max(corners, key=lambda c: c["y"])

        if target_node is None:
            return None, None # Không có đối tượng mục tiêu

        # Tính toán sai số
        target_x = target_node["x"]
        target_y = target_node["y"]
        
        error_x = target_x - self.anchor_x
        dy = self.anchor_y - target_y
        if dy <= 0: dy = 1  # Tránh chia cho 0 hoặc âm

        # Tính góc radian
        angle_rad = math.atan(error_x / dy)
        
        # Mapping góc sang dải giá trị lái (giả sử góc max là ~45 độ -> pi/4)
        # Giá trị trả về từ -1.0 (trái) đến 1.0 (phải)
        max_angle = math.pi / 4
        steering_val = angle_rad / max_angle
        
        # Clamp giá trị -1.0 đến 1.0
        steering_val = max(-1.0, min(1.0, steering_val))

        return self.base_speed, steering_val


class TurnModule:
    """
    Module xử lý rẽ bẻ góc tối đa, full tốc độ trong 2.5s.
    Sử dụng State Machine (Non-blocking) để không làm treo vòng lặp ảnh camera.
    """
    def __init__(self, img_width=640, turn_duration=1.6, max_speed=1.0, max_steering=1.0):
        self.img_width = img_width
        self.turn_duration = turn_duration
        self.max_speed = max_speed
        self.max_steering = max_steering

        self.is_turning = False
        self.turn_start_time = 0
        self.current_direction = None

    def trigger_turn_if_needed(self, detections):
        """
        Kiểm tra xem có cần kích hoạt rẽ không.
        Quy tắc:
        - Tại Interact Node có kèm biển báo cấm / rẽ trái / rẽ phải.
        - Tại Corner Node: Tự xác định hướng dựa vào vị trí corner 
          (Corner ở nửa trái màn hình -> rẽ phải, và ngược lại).
        """
        if self.is_turning:
            return False # Đang rẽ rồi thì không trigger lại

        corners = [d for d in detections if d["label"] == "corner"]
        interacts = [d for d in detections if d["label"] == "interact"]
        signs = [d for d in detections if d["label"] in ["turn_left", "turn_right", "forbidden"]]

        turn_dir = None

        # 1. Ưu tiên kiểm tra rẽ tại interact node + biển báo
        if len(interacts) > 0:
            for sign in signs:
                if sign["label"] == "turn_left":
                    turn_dir = "left"
                    break
                elif sign["label"] == "turn_right":
                    turn_dir = "right"
                    break
        
        # 2. Nếu không có interact node + biển báo, xét rẽ tại corner node
        if not turn_dir and len(corners) > 0:
            closest_corner = max(corners, key=lambda c: c["y"])
            if closest_corner["x"] < (self.img_width / 2):
                turn_dir = "right"  # Corner bên trái thì đường rẽ sang phải
            else:
                turn_dir = "left"   # Corner bên phải thì đường rẽ sang trái

        if turn_dir:
            self.start_turn(turn_dir)
            return True
            
        return False

    def start_turn(self, direction):
        self.is_turning = True
        self.turn_start_time = time.time()
        self.current_direction = direction
        print(f"[TURN MODULE] Bắt đầu rẽ {direction.upper()} trong {self.turn_duration}s")

    def process(self):
        """
        Hàm này cần được gọi liên tục trong vòng lặp while True của luồng chính.
        Trả về (speed, steering) nếu đang rẽ, trả về (None, None) nếu đã rẽ xong hoặc không rẽ.
        """
        if not self.is_turning:
            return None, None

        elapsed_time = time.time() - self.turn_start_time
        if elapsed_time >= self.turn_duration:
            # Đã hết thời gian rẽ
            print("[TURN MODULE] Kết thúc rẽ.")
            self.is_turning = False
            self.current_direction = None
            return None, None
        
        # Vẫn đang trong thời gian rẽ -> trả về max speed & max steering
        speed = self.max_speed
        steering = -self.max_steering if self.current_direction == "left" else self.max_steering
        
        return speed, steering

class DecisionModule:
    """
    Module ra quyết định hướng đi và tính toán lái dựa trên các node và biển báo.
    """
    def __init__(self, img_width=640, img_height=480, base_speed=0.3):
        self.img_width = img_width
        self.img_height = img_height
        self.base_speed = base_speed

        # Tọa độ neo (anchor) - tâm dưới cùng của ảnh
        self.anchor_x = self.img_width / 2
        self.anchor_y = self.img_height

    def update_resolution(self, width, height):
        self.img_width = width
        self.img_height = height
        self.anchor_x = self.img_width / 2
        self.anchor_y = self.img_height

    def make_decision(self, detections):
        """
        Input: list of detections, e.g., [{"label": "decision", "x": 320, "y": 400}, ...]
        Labels: decision, interact, corner, forbidden (cấm), turn_left, turn_right
        Output: (action, target_node, speed, steering) 
                action có thể là: "straight", "turn_left", "turn_right"
        """
        # Phân loại detections
        decisions = [d for d in detections if d["label"] == "decision"]
        interacts = [d for d in detections if d["label"] == "interact"]
        corners = [d for d in detections if d["label"] == "corner"]
        
        signs = {d["label"]: d for d in detections if d["label"] in ["forbidden", "turn_left", "turn_right"]}
        
        # 1. Thu thập tất cả các "node" có thể dùng để ra quyết định
        nodes = decisions + interacts + corners
        if not nodes:
            return "straight", None, 0.0, 0.0 # Mặc định đi thẳng chậm hoặc dừng nếu không có thông tin
            
        # 2. Ưu tiên node gần nhất (có tọa độ y lớn nhất do y=0 ở đỉnh ảnh)
        closest_node = max(nodes, key=lambda n: n["y"])
        label = closest_node["label"]
        
        action = "straight"
        
        # 3. Logic ra quyết định dựa vào loại node gần nhất
        if label == "interact":
            if "turn_left" in signs:
                action = "turn_left"
            elif "turn_right" in signs:
                action = "turn_right"
            elif "forbidden" in signs:
                # Gặp cấm ở interact node -> quyết định rẽ dựa trên đặc điểm node (vd vị trí x)
                # Giả sử interact node lệch trái -> ngã rẽ ở phải, nên rẽ phải; lệch phải -> rẽ trái
                if closest_node["x"] < (self.img_width / 2):
                    action = "turn_right"
                else:
                    action = "turn_left"
            else:
                action = "straight"
                
        elif label == "corner":
            # Tại corner, dựa trên ước lượng khoảng cách (y) và đặc điểm corner
            distance_estimated = self.img_height - closest_node["y"]
            
            # Kiểm tra xem xe đã đến đủ gần corner chưa (ngưỡng 40% chiều cao ảnh)
            if distance_estimated < (self.img_height * 0.4): 
                if closest_node["x"] < (self.img_width / 2):
                    action = "turn_right" # Corner bên trái -> đường rẽ ở bên phải
                else:
                    action = "turn_left" # Corner bên phải -> đường rẽ ở bên trái
            else:
                # Nếu còn xa corner thì vẫn giữ hướng đi thẳng về phía nó
                action = "straight"
                
        elif label == "decision":
            action = "straight"
            
        # 4. Tính toán góc lái (steering) nếu action là đi thẳng
        speed = self.base_speed
        steering = 0.0
        
        if action == "straight":
            target_x = closest_node["x"]
            target_y = closest_node["y"]
            
            error_x = target_x - self.anchor_x
            dy = self.anchor_y - target_y
            if dy <= 0: dy = 1
            
            angle_rad = math.atan(error_x / dy)
            max_angle = math.pi / 4
            steering_val = angle_rad / max_angle
            steering = max(-1.0, min(1.0, steering_val))
            
        return action, closest_node, speed, steering
