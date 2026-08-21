import socket
import struct
import json
import cv2

class RemoteYOLOClient:
    def __init__(self, server_ip, port=5000):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[RemoteYOLO] Đang kết nối tới Laptop tại {server_ip}:{port}...")
        try:
            self.client_socket.connect((server_ip, port))
            print("[RemoteYOLO] Đã kết nối thành công!")
        except Exception as e:
            print(f"[RemoteYOLO] Lỗi kết nối: {e}")
            raise e

    def _recvall(self, count):
        buf = b''
        while count:
            newbuf = self.client_socket.recv(count)
            if not newbuf: 
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def get_detections(self, frame):
        """
        Gửi frame tới server và nhận lại list detections dạng JSON.
        Mỗi detection là một dict: {"label": "...", "x": ..., "y": ..., "id": ...}
        """
        # Nén ảnh
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        result, img_encoded = cv2.imencode('.jpg', frame, encode_param)
        data_bytes = img_encoded.tobytes()
        
        # Gửi
        self.client_socket.sendall(struct.pack('<I', len(data_bytes)))
        self.client_socket.sendall(data_bytes)
        
        # Nhận
        lengthbuf = self._recvall(4)
        if not lengthbuf: 
            return []
        length, = struct.unpack('<I', lengthbuf)
        
        json_bytes = self._recvall(length)
        if not json_bytes: 
            return []
        
        return json.loads(json_bytes.decode('utf-8'))
        
    def draw_detections(self, frame, detections):
        """Vẽ detections lên frame để debug"""
        for d in detections:
            label_text = f"{d['label']} {d.get('id', '')}"
            cv2.putText(frame, label_text, (int(d['x']), int(d['y'])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (int(d['x']), int(d['y'])), 4, (0,0,255), -1)
        return frame

    def close(self):
        self.client_socket.close()
