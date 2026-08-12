from abc import ABC, abstractmethod

class BaseLidarProcessor(ABC):
    """Lớp cơ sở cho xử lý dữ liệu từ LiDAR."""

    @abstractmethod
    def initialize(self):
        """Khởi tạo cảm biến LiDAR."""
        pass

    @abstractmethod
    def get_scan(self):
        """Lấy dữ liệu quét từ LiDAR."""
        pass

    @abstractmethod
    def process_scan(self, scan_data):
        """Xử lý dữ liệu quét (ví dụ: lọc nhiễu, tìm điểm gần nhất).
        Args:
            scan_data: Dữ liệu trả về từ LiDAR (thường là mảng khoảng cách và góc).
        Returns:
            Thông tin trích xuất được (như cảnh báo va chạm).
        """
        pass

    @abstractmethod
    def process(self, blackboard):
        """Đọc và ghi kết quả xử lý vào blackboard."""
        pass
