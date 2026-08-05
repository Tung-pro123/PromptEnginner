import math
from utils.config_loader import load_yaml

class RoutePlanner:
    def __init__(self, map_config_path="d:/AI_Project/racing_promax/catkin_ws/src/jetracer_smartcity/config/intersection_map.yaml"):
        # Load map
        config = load_yaml(map_config_path)
        self.map_cfg = config["map"]
        self.start_node = self.map_cfg["start_node"]
        self.finish_node = self.map_cfg["finish_node"]
        
        # Load graph topology
        self.nodes = {}
        for node_id, data in self.map_cfg["nodes"].items():
            self.nodes[node_id] = {
                "x": data["x"],
                "y": data["y"],
                "connections": list(data["connections"])
            }
            
        # Keep track of blocked connections (dynamic obstacles or prohibited turns)
        # Structure: Set of tuples (from_node, to_node)
        self.blocked_edges = set()

    def block_edge(self, from_node, to_node):
        """
        Dynamically block an edge (e.g., when a prohibitory sign 'no_left' is detected).
        """
        self.blocked_edges.add((from_node, to_node))
        print(f"[RoutePlanner] Blocked edge: {from_node} -> {to_node}")

    def unblock_edge(self, from_node, to_node):
        """
        Unblock an edge.
        """
        if (from_node, to_node) in self.blocked_edges:
            self.blocked_edges.remove((from_node, to_node))
            print(f"[RoutePlanner] Unblocked edge: {from_node} -> {to_node}")

    def reset_blocks(self):
        """
        Resets all blocked edges.
        """
        self.blocked_edges.clear()

    def find_path(self, start_node, target_node):
        """
        BFS-based shortest path finder that respects blocked edges.
        Returns: list of node IDs from start to target. Returns None if path doesn't exist.
        """
        if start_node not in self.nodes or target_node not in self.nodes:
            return None
            
        queue = [[start_node]]
        visited = {start_node}
        
        while queue:
            path = queue.pop(0)
            node = path[-1]
            
            if node == target_node:
                return path
                
            for neighbor in self.nodes[node]["connections"]:
                # Check if this edge is blocked
                if (node, neighbor) in self.blocked_edges:
                    continue
                    
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                    
        return None

    def get_required_direction(self, prev_node, curr_node, next_node):
        """
        Computes the relative turn direction ('straight', 'left', 'right') at 'curr_node' 
        given coordinates of 'prev_node', 'curr_node', and 'next_node'.
        """
        if curr_node not in self.nodes or next_node not in self.nodes:
            return "straight"
            
        # Coordinates
        curr_x, curr_y = self.nodes[curr_node]["x"], self.nodes[curr_node]["y"]
        next_x, next_y = self.nodes[next_node]["x"], self.nodes[next_node]["y"]
        
        # Calculate next vector
        v_next_x = next_x - curr_x
        v_next_y = next_y - curr_y
        
        if prev_node is None:
            # At start, assume default heading is straight North relative to coordinate axis (0, 1)
            v_curr_x = 0.0
            v_curr_y = 1.0
        else:
            prev_x, prev_y = self.nodes[prev_node]["x"], self.nodes[prev_node]["y"]
            v_curr_x = curr_x - prev_x
            v_curr_y = curr_y - prev_y
            
        # Normalize vectors for reliability
        len_curr = math.sqrt(v_curr_x**2 + v_curr_y**2)
        len_next = math.sqrt(v_next_x**2 + v_next_y**2)
        
        if len_curr == 0 or len_next == 0:
            return "straight"
            
        v_curr_x /= len_curr
        v_curr_y /= len_curr
        v_next_x /= len_next
        v_next_y /= len_next
        
        # Cross product (Z-component in 3D: Ax * By - Ay * Bx)
        # Represents direction of rotation: > 0 for Left, < 0 for Right (assuming standard Cartesian coordinate system)
        cross_prod = v_curr_x * v_next_y - v_curr_y * v_next_x
        
        # Dot product: Ax * Bx + Ay * By
        # Represents forward/backward alignment
        dot_prod = v_curr_x * v_next_x + v_curr_y * v_next_y
        
        # Thresholds to account for float inaccuracy
        epsilon = 0.1
        
        if dot_prod > 0.7:  # Angles less than ~45 degrees
            return "straight"
        elif dot_prod < -0.7: # U-turn
            return "u_turn"
        elif cross_prod > epsilon:
            return "left"
        elif cross_prod < -epsilon:
            return "right"
            
        return "straight"
