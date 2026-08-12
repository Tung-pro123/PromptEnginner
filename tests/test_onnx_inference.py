import os
import sys
import numpy as np

# Đảm bảo đường dẫn gốc của project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Đường dẫn trỏ tới model ONNX
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "vision_inference.onnx")

def test_onnx_inference():
    print(f"=== TEST CHẠY SUY LUẬN (INFERENCE) VỚI ONNXRUNTIME ===")
    print(f"[INFO] Tìm model tại: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] Không tìm thấy file model. Vui lòng kiểm tra lại!")
        return

    # [FIX JETSON] Tự động thêm đường dẫn CUDA vào môi trường trước khi gọi ONNXRuntime
    if "/usr/local/cuda/lib64" not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

    try:
        import onnxruntime as ort
        print(f"[OK] Đã import onnxruntime phiên bản {ort.__version__}")
    except ImportError:
        print("[LỖI] Chưa cài đặt onnxruntime. Vui lòng chạy: pip install onnxruntime-gpu")
        return

    try:
        # Load model với các Provider phù hợp cho Jetson
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
        print(f"[OK] Đã load model thành công!")
        print(f"[INFO] Backend đang chạy: {session.get_providers()}")
        
        # Lấy tên input
        input_name = session.get_inputs()[0].name
        
        # Tạo dữ liệu giả (Dummy Data) giống với ảnh đầu vào của model
        # Shape: (Batch_size=1, Channel=1, Height=32, Width=128)
        # Type: np.float32, giá trị từ 0.0 -> 1.0 (như ảnh đã normalize)
        dummy_input = np.random.rand(1, 1, 32, 128).astype(np.float32)
        print(f"[INFO] Đã tạo Dữ liệu giả (Dummy Input) với shape: {dummy_input.shape}")
        
        # Chạy inference
        print(f"\n[INFO] Đang chạy Inference...")
        outputs = session.run(None, {input_name: dummy_input})
        
        # In kết quả
        # Model của bạn trả về 1 tensor có shape (1, 2) chứa [steer, throttle]
        steer = float(outputs[0][0][0])
        throttle = float(outputs[0][0][1])
        
        print(f"[OK] Chạy Inference thành công!")
        print(f"----------------------------------------")
        print(f"🔹 Kết quả Output (Steer): {steer:+.4f}")
        print(f"🔹 Kết quả Output (Throttle): {throttle:+.4f}")
        print(f"----------------------------------------")
        print("=> Model ONNX của bạn đã sẵn sàng để gắn vào xe chạy thực tế!")

    except Exception as e:
        print(f"[LỖI] Quá trình chạy bị lỗi: {e}")

if __name__ == "__main__":
    test_onnx_inference()
