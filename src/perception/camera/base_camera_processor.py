from abc import ABC, abstractmethod

class BaseCameraProcessor(ABC):
    """Lớp cơ sở cho xử lý dữ liệu từ camera."""

    @abstractmethod
    def initialize(self):
        """Khởi tạo camera và các cấu hình liên quan."""
        pass

    @abstractmethod
    def get_frame(self):
        """Lấy frame ảnh hiện tại từ camera."""
        pass

    @abstractmethod
    def process_frame(self, frame):
        """Xử lý frame ảnh (ví dụ: tracking vạch, nhận diện vật cản).
        Args:
            frame: Ảnh numpy array (từ OpenCV).
        Returns:
            Kết quả sau xử lý (ví dụ: góc lệch, danh sách vật cản).
        """
        pass
