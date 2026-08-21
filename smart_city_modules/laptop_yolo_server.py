import socket
import cv2
import numpy as np
import struct
import json
from ultralytics import YOLO

# Cấu hình IP và Port (0.0.0.0 nghĩa là chấp nhận kết nối từ mọi IP trong mạng)
HOST = '10.3.91.206'
PORT = 5000

def recvall(sock, count):
    """Hàm hỗ trợ để nhận đủ số byte dữ liệu qua TCP"""
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: 
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def start_server():
    # Load model YOLO (chạy trên Laptop)
    print("Đang khởi động model YOLO trên Laptop...")
    model = YOLO("../models/yolo.pt")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f"\n[SERVER] Đã mở cổng {PORT}. Đang chờ Jetson kết nối...")
    
    while True:
        conn, addr = server_socket.accept()
        print(f"\n[SERVER] Jetson đã kết nối từ IP: {addr}")
        
        try:
            while True:
                # 1. Nhận thông tin kích thước ảnh (4 bytes)
                lengthbuf = recvall(conn, 4)
                if not lengthbuf: 
                    break
                length, = struct.unpack('<I', lengthbuf)
                
                # 2. Nhận dữ liệu ảnh (đã nén JPEG)
                img_data = recvall(conn, length)
                if not img_data: 
                    break
                
                # Giải mã ảnh JPEG sang Frame OpenCV
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                # 3. Chạy Tracking YOLO
                results = model.track(frame, conf=0.25, persist=True, verbose=False)
                r = results[0]
                
                # Lọc box ảo để hiển thị trên màn hình Laptop sạch sẽ hơn
                if len(r.boxes) > 0:
                    keep_indices = list(range(len(r.boxes)))
                    dec_cls_idx = next((k for k, v in model.names.items() if v == "Decision"), None)
                    int_cls_idx = next((k for k, v in model.names.items() if v == "Interact"), None)
                    cor_cls_idx = next((k for k, v in model.names.items() if v == "Corner"), None)
                    
                    dec_indices = [i for i in keep_indices if int(r.boxes.cls[i].item()) == dec_cls_idx]
                    if dec_indices:
                        best_dec_i = max(dec_indices, key=lambda i: r.boxes.xywh[i][1].item())
                        y_dec = r.boxes.xywh[best_dec_i][1].item()
                        for i in [idx for idx in keep_indices if int(r.boxes.cls[idx].item()) in (int_cls_idx, cor_cls_idx)]:
                            if r.boxes.xywh[i][1].item() > y_dec + 15:
                                keep_indices.remove(i)
                    r = r[keep_indices]
                
                # 4. Đóng gói kết quả detections
                detections = []
                if len(r.boxes) > 0:
                    ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
                    
                    # Ưu tiên lấy masks nếu có (segmentation)
                    if r.masks is not None:
                        for box, mask, cls, track_id in zip(r.boxes, r.masks, r.boxes.cls, ids):
                            label = model.names[int(cls)]
                            xy = mask.xy[0]
                            if len(xy) > 0:
                                x, y = np.mean(xy[:, 0]), np.mean(xy[:, 1])
                                detections.append({"label": label, "x": float(x), "y": float(y), "id": track_id})
                    else:
                        for box, cls, track_id in zip(r.boxes, r.boxes.cls, ids):
                            label = model.names[int(cls)]
                            x, y, w, h = box.xywh[0]
                            detections.append({"label": label, "x": float(x), "y": float(y), "id": track_id})
                            
                # 5. Gửi kết quả JSON về lại cho Jetson
                data_str = json.dumps(detections)
                data_bytes = data_str.encode('utf-8')
                
                conn.sendall(struct.pack('<I', len(data_bytes))) # Gửi độ dài chuỗi JSON
                conn.sendall(data_bytes)                         # Gửi nội dung JSON
                
                # Hiển thị trên màn hình Laptop
                annotated_frame = r.plot()
                cv2.imshow("Laptop YOLO Server - Live View", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Đã nhận phím Q. Đóng server...")
                    return
                    
        except Exception as e:
            print(f"[SERVER] Lỗi trong quá trình truyền dữ liệu: {e}")
        finally:
            conn.close()
            print("[SERVER] Jetson đã ngắt kết nối. Sẵn sàng nhận kết nối mới...")

if __name__ == '__main__':
    start_server()
