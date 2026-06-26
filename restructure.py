import os
import shutil

def main():
    base_dir = r"d:\Jetson\Jetson"
    
    # 1. Create new directories
    dirs_to_create = [
        "archive",
        "docs",
        "src/core/perception",
        "src/core/control",
        "src/core/planning",
        "src/core/utils",
        "src/speed_track",
        "src/smart_city"
    ]
    for d in dirs_to_create:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    
    # 2. Move problem_* to archive
    for prob in ["problem_a", "problem_b", "problem_c"]:
        src = os.path.join(base_dir, prob)
        dst = os.path.join(base_dir, "archive", prob)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"Moved {prob} to archive.")
            
    # 3. Move context to docs
    context_dir = os.path.join(base_dir, "context")
    docs_dir = os.path.join(base_dir, "docs")
    if os.path.exists(context_dir):
        for item in os.listdir(context_dir):
            shutil.move(os.path.join(context_dir, item), os.path.join(docs_dir, item))
        os.rmdir(context_dir)
        print("Moved context files to docs.")

    # 4. Copy shared files to src/core
    shared_files = {
        "map_navigator.py": "src/core/planning/map_navigator.py",
        "callmap.py": "src/core/planning/callmap.py",
        "opposite_detector.py": "src/core/utils/opposite_detector.py",
        "map.json": "src/core/utils/map.json"
    }
    
    prob_a_archive = os.path.join(base_dir, "archive", "problem_a")
    for f_src, f_dst in shared_files.items():
        src_path = os.path.join(prob_a_archive, f_src)
        dst_path = os.path.join(base_dir, f_dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {f_src} to {f_dst}")
            
    # Modify map.json path in callmap.py
    callmap_path = os.path.join(base_dir, "src/core/planning/callmap.py")
    if os.path.exists(callmap_path):
        with open(callmap_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace('"map.json"', '"../utils/map.json"')
        with open(callmap_path, 'w', encoding='utf-8') as f:
            f.write(content)

    # Modify map.json path in map_navigator.py
    mapnav_path = os.path.join(base_dir, "src/core/planning/map_navigator.py")
    if os.path.exists(mapnav_path):
        with open(mapnav_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace('json_path="map.json"', 'json_path=os.path.join(os.path.dirname(__file__), "..", "utils", "map.json")')
        content = "import os\n" + content
        with open(mapnav_path, 'w', encoding='utf-8') as f:
            f.write(content)

    # 5. Create main_speed_track.py
    prob_a_file = os.path.join(prob_a_archive, "problem_a.py")
    speed_main = os.path.join(base_dir, "src/speed_track/main_speed_track.py")
    if os.path.exists(prob_a_file):
        with open(prob_a_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add path fix
        path_fix = "import sys\nimport os\nsys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))\n"
        content = content.replace("import rospy", path_fix + "import rospy", 1)
        
        # Fix imports
        content = content.replace("from opposite_detector import SimpleOppositeDetector", "from src.core.utils.opposite_detector import SimpleOppositeDetector")
        content = content.replace("from map_navigator import MapNavigator", "from src.core.planning.map_navigator import MapNavigator")
        content = content.replace('self.MAP_FILE_PATH = "map.json"', 'self.MAP_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "core", "utils", "map.json")')
        
        with open(speed_main, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Created main_speed_track.py")

    # 6. Create main_smart_city.py
    prob_b_archive = os.path.join(base_dir, "archive", "problem_b")
    prob_b_file = os.path.join(prob_b_archive, "problem_b.py")
    smart_main = os.path.join(base_dir, "src/smart_city/main_smart_city.py")
    if os.path.exists(prob_b_file):
        with open(prob_b_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add path fix
        path_fix = "import sys\nimport os\nsys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))\n"
        content = content.replace("import rospy", path_fix + "import rospy", 1)
        
        # Fix imports
        content = content.replace("from opposite_detector import SimpleOppositeDetector", "from src.core.utils.opposite_detector import SimpleOppositeDetector")
        content = content.replace("from map_navigator import MapNavigator", "from src.core.planning.map_navigator import MapNavigator")
        content = content.replace('self.MAP_FILE_PATH = "map.json"', 'self.MAP_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "core", "utils", "map.json")')
        
        with open(smart_main, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Created main_smart_city.py")
        
    print("Restructuring completed successfully!")

if __name__ == "__main__":
    main()
