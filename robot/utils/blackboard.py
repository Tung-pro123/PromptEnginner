class Blackboard:
    """
    Bảng đen (Blackboard) lưu trữ dữ liệu trung tâm.
    Tất cả các Knowledge Sources (Processors) sẽ đọc và ghi dữ liệu qua đây.
    """
    def __init__(self):
        self._data = {}

    def set(self, key, value):
        """Lưu một giá trị vào Blackboard."""
        self._data[key] = value

    def get(self, key, default=None):
        """Lấy một giá trị từ Blackboard."""
        return self._data.get(key, default)

    def has(self, key):
        """Kiểm tra xem dữ liệu có tồn tại trong Blackboard không."""
        return key in self._data

    def remove(self, key):
        """Xóa một dữ liệu khỏi Blackboard."""
        if key in self._data:
            del self._data[key]

    def dump(self):
        """Lấy toàn bộ dữ liệu (thường dùng để debug)."""
        return self._data.copy()
