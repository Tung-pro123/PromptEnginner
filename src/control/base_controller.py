from abc import ABC, abstractmethod

class BaseController(ABC):
    """Lớp cơ sở cho logic điều khiển robot."""

    @abstractmethod
    def initialize(self):
        """Khởi tạo các thành phần điều khiển (động cơ, giao tiếp, v.v.)."""
        pass

    @abstractmethod
    def move(self, speed, direction):
        """Ra lệnh di chuyển robot.
        Args:
            speed: Tốc độ di chuyển.
            direction: Hướng di chuyển hoặc góc quay.
        """
        pass

    @abstractmethod
    def stop(self):
        """Dừng xe."""
        pass

    @abstractmethod
    def process(self, blackboard):
        """Đọc và ghi kết quả xử lý vào blackboard."""
        pass
