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
        corners = [d for d in detections if d["label"] == "Corner"]
        decisions = [d for d in detections if d["label"] == "Decision"]
        
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

        corners = [d for d in detections if d["label"] == "Corner"]
        interacts = [d for d in detections if d["label"] == "Interact"]
        signs = [d for d in detections if d["label"] in ["turn_left", "turn_right", "Forbidden"]]

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
        Output: (action, target_node, speed, steering, steps) 
                action có thể là: "straight", "turn_left", "turn_right"
                steps là list các chuỗi mô tả quá trình suy luận
        """
        steps = []
        
        # Phân loại detections
        decisions = [d for d in detections if d["label"] == "Decision"]
        interacts = [d for d in detections if d["label"] == "Interact"]
        corners = [d for d in detections if d["label"] == "Corner"]
        
        signs = {d["label"]: d for d in detections if d["label"] in ["Forbidden", "turn_left", "turn_right"]}
        if signs:
            steps.append(f"Nhận diện biển báo: {list(signs.keys())}")
        
        # 1. Thu thập tất cả các "node" có thể dùng để ra quyết định
        nodes = decisions + interacts + corners
        if not nodes:
            steps.append("Không thấy node nào -> mặc định đi thẳng chậm (dừng).")
            return "straight", None, 0.0, 0.0, steps
            
        steps.append(f"Tìm thấy {len(nodes)} nodes. Đang chọn node gần nhất.")
        
        # 2. Ưu tiên node gần nhất (có tọa độ y lớn nhất do y=0 ở đỉnh ảnh)
        closest_node = max(nodes, key=lambda n: n["y"])
        label = closest_node["label"]
        steps.append(f"Node gần nhất là: {label} (x={closest_node['x']:.0f}, y={closest_node['y']:.0f})")
        
        action = "straight"
        
        # 3. Logic ra quyết định dựa vào loại node gần nhất
        if label == "Interact":
            if "turn_left" in signs:
                action = "turn_left"
                steps.append("Đang ở interact node + gặp biển rẽ trái -> Quyết định Rẽ Trái.")
            elif "turn_right" in signs:
                action = "turn_right"
                steps.append("Đang ở interact node + gặp biển rẽ phải -> Quyết định Rẽ Phải.")
            elif "Forbidden" in signs:
                if closest_node["x"] < (self.img_width / 2):
                    action = "turn_right"
                    steps.append("Interact node lệch trái + gặp biển CẤM -> Quyết định Rẽ Phải.")
                else:
                    action = "turn_left"
                    steps.append("Interact node lệch phải + gặp biển CẤM -> Quyết định Rẽ Trái.")
            else:
                action = "straight"
                steps.append("Interact node không có biển báo -> Quyết định Đi Thẳng.")
                
        elif label == "Corner":
            distance_estimated = self.img_height - closest_node["y"]
            steps.append(f"Đang ở corner. Khoảng cách ước lượng (theo Y): {distance_estimated:.0f}px")
            
            if distance_estimated < (self.img_height * 0.4): 
                if closest_node["x"] < (self.img_width / 2):
                    action = "turn_right"
                    steps.append(f"Đã đủ gần corner (lệch trái) -> Quyết định Rẽ Phải.")
                else:
                    action = "turn_left"
                    steps.append(f"Đã đủ gần corner (lệch phải) -> Quyết định Rẽ Trái.")
            else:
                action = "straight"
                steps.append("Corner còn xa -> Quyết định Đi Thẳng hướng tới corner.")
                
        elif label == "Decision":
            action = "straight"
            steps.append("Đang bám theo decision node -> Quyết định Đi Thẳng.")
            
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
            steps.append(f"Tính toán Steering: ErrorX={error_x:.0f}, dy={dy:.0f} => Góc lái: {steering:.2f}")
            
        return action, closest_node, speed, steering, steps
