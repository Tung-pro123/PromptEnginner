import os
import sys

# Đảm bảo đường dẫn gốc của project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Đường dẫn mặc định trỏ tới một model onnx có sẵn trong thư mục models (ví dụ yolov5n.onnx hoặc dagger_policy.onnx)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "yolov5n.onnx")

def test_onnxruntime():
    print("1. Kiểm tra bằng ONNXRuntime...")
    try:
        import onnxruntime as ort
        print(f"   [OK] Đã import onnxruntime phiên bản {ort.__version__}")
        
        if os.path.exists(MODEL_PATH):
            # Ưu tiên load bằng TensorRT hoặc CUDA trên Jetson
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(MODEL_PATH, providers=providers)
            print(f"   [OK] Đã load model thành công!")
            print(f"   [INFO] Các Provider (Backend) đang kích hoạt: {session.get_providers()}")
            
            inputs = session.get_inputs()
            for idx, i in enumerate(inputs):
                print(f"   [INFO] Input {idx}: name='{i.name}', shape={i.shape}, type={i.type}")
        else:
            print(f"   [CẢNH BÁO] Không tìm thấy file model để test tại: {MODEL_PATH}")
    except ImportError:
        print("   [LỖI] Chưa cài đặt onnxruntime. Vui lòng chạy: pip install onnxruntime hoặc onnxruntime-gpu")
    except Exception as e:
        print(f"   [LỖI] Quá trình chạy bị lỗi: {e}")

def test_opencv_dnn():
    print("\n2. Kiểm tra bằng OpenCV DNN...")
    try:
        import cv2
        print(f"   [OK] Đã import OpenCV phiên bản {cv2.__version__}")
        
        if os.path.exists(MODEL_PATH):
            net = cv2.dnn.readNetFromONNX(MODEL_PATH)
            
            # Cấu hình sử dụng CUDA (phù hợp với Jetson)
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("   [OK] Đã thiết lập OpenCV DNN chạy bằng backend CUDA.")
            except Exception:
                print("   [INFO] OpenCV hiện tại không được build kèm CUDA backend. Sẽ chạy bằng CPU.")
                
            print("   [OK] Đã load model bằng cv2.dnn thành công!")
        else:
            print(f"   [CẢNH BÁO] Không tìm thấy file model tại: {MODEL_PATH}")
    except ImportError:
        print("   [LỖI] Chưa cài đặt OpenCV.")
    except Exception as e:
        print(f"   [LỖI] Quá trình load bằng OpenCV bị lỗi: {e}")

def test_tensorrt():
    print("\n3. Kiểm tra module TensorRT (Đặc thù của Jetson/Nvidia)...")
    try:
        import tensorrt as trt
        print(f"   [OK] Đã import TensorRT phiên bản {trt.__version__} thành công!")
    except ImportError:
        print("   [INFO] Không tìm thấy TensorRT binding cho Python (Điều này bình thường nếu bạn sử dụng ONNXRuntime/OpenCV).")

if __name__ == "__main__":
    print("=== CHƯƠNG TRÌNH KIỂM TRA IMPORT MODEL AI TRÊN JETSON ===\n")
    test_onnxruntime()
    test_opencv_dnn()
    test_tensorrt()
    print("\n=== HOÀN TẤT KIỂM TRA ===")
