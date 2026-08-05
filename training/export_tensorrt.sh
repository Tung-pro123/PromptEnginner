#!/bin/bash
# Convert ONNX model to TensorRT FP16 engine on the Jetson Nano.
# Ensure this script is run on the Jetson itself, as TRT engines are hardware-dependent.

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_onnx_model> <path_to_output_engine>"
    echo "Example: $0 yolov5n_smartcity.onnx yolov5n_smartcity.engine"
    exit 1
fi

ONNX_PATH=$1
ENGINE_PATH=$2

# Check if trtexec exists
if ! command -v trtexec &> /dev/null
then
    echo "trtexec command could not be found. Searching in default Jetpack paths..."
    if [ -f /usr/src/tensorrt/bin/trtexec ]; then
        TRTEXEC_BIN=/usr/src/tensorrt/bin/trtexec
    else
        echo "Error: trtexec not found. Make sure TensorRT is installed and added to PATH."
        echo "Typically: export PATH=/usr/src/tensorrt/bin:\$PATH"
        exit 1
    fi
else
    TRTEXEC_BIN=trtexec
fi

echo "Starting conversion of $ONNX_PATH to $ENGINE_PATH using FP16..."
$TRTEXEC_BIN \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --fp16 \
  --workspace=1024 \
  --verbose

if [ $? -eq 0 ]; then
    echo "Success! TensorRT engine generated at: $ENGINE_PATH"
else
    echo "Error: TensorRT compilation failed."
    exit 1
fi
