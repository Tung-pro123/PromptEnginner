import argparse
import sys
import torch

def export_to_onnx(weights_path, output_path, img_size=320):
    """
    Exports a PyTorch model to ONNX format (Opset 11).
    Optimized for compatibility with TensorRT 7.x on Jetpack 4.5.1.
    """
    print(f"Loading weights from {weights_path}...")
    try:
        # Load YOLOv5 model using torch.hub or custom loader
        # Since this script runs on host/Colab, we attempt to load using torch.hub
        # fallback to standard model loading if hub fails
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
    except Exception as e:
        print(f"Failed to load via torch.hub: {e}")
        print("Please ensure you run this script inside the yolov5/ repository directory.")
        return False

    model.eval()
    
    # Create dummy input tensor matching model input dimensions (BCHW)
    dummy_input = torch.zeros(1, 3, img_size, img_size)
    
    # Dry run
    try:
        _ = model(dummy_input)
    except Exception as e:
        print(f"Dry run failed: {e}")
        return False

    print(f"Exporting to ONNX format (opset 11) at: {output_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            verbose=False,
            opset_version=11,  # Essential for TensorRT 7.1 support
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=None  # Static shapes are faster and more stable on older TensorRT
        )
        print("ONNX export completed successfully.")
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX Opset 11")
    parser.add_argument("--weights", type=str, required=True, help="Path to PyTorch .pt weights file")
    parser.add_argument("--output", type=str, default="yolov5n_smartcity.onnx", help="Path to output .onnx file")
    parser.add_argument("--imgsz", type=int, default=320, help="Model input image size (square)")
    
    args = parser.parse_known_args()[0]
    export_to_onnx(args.weights, args.output, args.imgsz)
