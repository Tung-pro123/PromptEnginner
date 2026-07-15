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
        """Dừng robot khẩn cấp hoặc dừng hẳn."""
        pass
