from abc import ABC, abstractmethod

class BaseDataCollector(ABC):
    """
    Lớp cơ sở (Base Class) chung cho việc thu thập dữ liệu từ các cảm biến.
    Các cảm biến cụ thể (Camera, Lidar, IMU, v.v.) sẽ kế thừa lớp này.
    """

    def __init__(self, sensor_name):
        self.sensor_name = sensor_name
        self.is_connected = False

    @abstractmethod
    def connect(self):
        """Khởi tạo kết nối đến cảm biến."""
        pass

    @abstractmethod
    def read_data(self):
        """Đọc dữ liệu nguyên thủy (raw data) từ cảm biến.
        Returns:
            Dữ liệu trả về tùy thuộc vào loại cảm biến.
        """
        pass

    @abstractmethod
    def close(self):
        """Đóng kết nối và giải phóng tài nguyên."""
        pass

    def get_sensor_info(self):
        """Lấy thông tin trạng thái của cảm biến."""
        return {
            "sensor_name": self.sensor_name,
            "is_connected": self.is_connected
        }
