from abc import ABC, abstractmethod

class BaseDataWriter(ABC):
    """
    Lớp cơ sở (Base Class) chung cho việc ghi và lưu trữ dữ liệu (vào file CSV, JSON, HDF5, ROS bag...).
    """

    def __init__(self, output_path):
        self.output_path = output_path
        self.is_open = False

    @abstractmethod
    def open(self):
        """Mở kết nối tới file hoặc database."""
        pass

    @abstractmethod
    def write(self, data):
        """Ghi dữ liệu.
        Args:
            data: Dữ liệu cần ghi (có thể là dict, tuple, numpy array...).
        """
        pass

    @abstractmethod
    def close(self):
        """Đóng file/database."""
        pass
