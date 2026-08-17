import os
import sys
import numpy as np

# Đảm bảo đường dẫn gốc của project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tìm model ONNX có sẵn trong thư mục models (ví dụ: yolov5n.onnx hoặc dagger_policy.onnx)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_MODEL = os.path.join(MODELS_DIR, "yolov5n.onnx")

def test_onnx_inference(model_path=DEFAULT_MODEL):
    print(f"=== TEST CHẠY SUY LUẬN (INFERENCE) VỚI ONNXRUNTIME ===")
    print(f"[INFO] Tìm model tại: {model_path}")
    
    if not os.path.exists(model_path):
        # Thử tìm file .onnx bất kỳ trong thư mục models
        onnx_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.onnx')]
        if onnx_files:
            model_path = os.path.join(MODELS_DIR, onnx_files[0])
            print(f"[INFO] Đã tự động chọn model: {model_path}")
        else:
            print(f"[LỖI] Không tìm thấy file .onnx nào trong thư mục models ({MODELS_DIR})!")
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
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"[OK] Đã load model thành công!")
        print(f"[INFO] Backend đang chạy: {session.get_providers()}")
        
        # Lấy thông tin input
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape
        
        # Xử lý dynamic batch size hoặc shape động
        concrete_shape = [dim if (isinstance(dim, int) and dim > 0) else 1 for dim in input_shape]
        dummy_input = np.random.rand(*concrete_shape).astype(np.float32)
        print(f"[INFO] Đã tạo Dữ liệu giả (Dummy Input) cho '{input_name}' với shape: {dummy_input.shape}")
        
        # Chạy inference
        print(f"\n[INFO] Đang chạy Inference...")
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"[OK] Chạy Inference thành công! Số outputs trả về: {len(outputs)}")
        for idx, out in enumerate(outputs):
            print(f"🔹 Output {idx} shape: {out.shape}, dtype: {out.dtype}")
        print("=> Model ONNX của bạn đã sẵn sàng để hoạt động!")

    except Exception as e:
        print(f"[LỖI] Quá trình chạy bị lỗi: {e}")

if __name__ == "__main__":
    test_onnx_inference()
