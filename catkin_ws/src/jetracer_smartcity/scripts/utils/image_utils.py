import cv2
import numpy as np

def letterbox(img, new_shape=(320, 320), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize image to a square shape while maintaining aspect ratio, padding with a border color.
    Used for YOLO preprocessing to prevent stretching of traffic signs.
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better performance/quality)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape[1], new_shape[0]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, left, top)

def preprocess_image(frame, target_size=(320, 320)):
    """
    Preprocess image frame: Letterbox, Convert BGR to RGB, Normalize [0, 1], and Transpose to BCHW.
    """
    # 1. Letterbox resize
    padded_img, ratio, pad_details = letterbox(frame, target_size, auto=False)
    
    # 2. BGR to RGB
    rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    
    # 3. HWC to CHW
    chw_img = rgb_img.transpose((2, 0, 1))
    
    # 4. Normalize to [0.0, 1.0] and add batch dimension (BCHW)
    normalized = np.ascontiguousarray(chw_img, dtype=np.float32) / 255.0
    batch_img = np.expand_dims(normalized, axis=0)
    
    return batch_img, ratio, pad_details
