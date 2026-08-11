"""
error_logger.py - Mô-đun ghi lỗi tập trung dùng chung cho toàn bộ tasks.

Tính năng:
  - In traceback đầy đủ ra console (stderr) ngay lập tức.
  - Đồng thời ghi log vào file crash_<task_name>.log trong thư mục logs/.
  - Tương thích với cả khi có ROS lẫn khi không có ROS.
"""
import sys
import os
import traceback
import datetime

def log_crash(task_name: str, exc: BaseException):
    """
    In chi tiết lỗi ra console và ghi vào file log.
    Gọi hàm này bên trong khối except.
    """
    tb_str = traceback.format_exc()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = (
        f"\n{'='*60}\n"
        f"[CRASH] {timestamp} | Task: {task_name}\n"
        f"Lỗi: {type(exc).__name__}: {exc}\n"
        f"{'='*60}\n"
        f"{tb_str}"
        f"{'='*60}\n"
    )

    # 1. In ra console (stderr) ngay lập tức để thấy trong terminal
    print(header, file=sys.stderr, flush=True)

    # 2. Thử in thêm qua rospy nếu có ROS
    try:
        import rospy
        rospy.logerr(f"[CRASH] Task={task_name} | {type(exc).__name__}: {exc}")
        rospy.logerr(tb_str)
    except Exception:
        pass  # ROS chưa init hoặc đã shutdown - không sao

    # 3. Ghi vào file logs/crash_<task_name>.log
    try:
        # Thư mục logs nằm cùng cấp với thư mục tasks (tức là repo root)
        repo_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(repo_root, "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"crash_{task_name}.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(header)
        print(f"[CRASH LOG] Đã ghi vào: {log_path}", file=sys.stderr, flush=True)
    except Exception as write_err:
        print(f"[CRASH LOG] Không thể ghi file log: {write_err}", file=sys.stderr)
