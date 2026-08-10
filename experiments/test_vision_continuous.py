import os
import pandas as pd
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import model và tham số từ file train
from train_vision_autoencoder import VisionAutoencoder, TARGETS, IMG_WIDTH, IMG_HEIGHT, LATENT_DIM

def test_continuous():
    dataset_dir = r"e:\robot-jeston\logs\dataset"
    images_dir = os.path.join(dataset_dir, "images")
    csv_path = os.path.join(dataset_dir, "dataset_labels.csv")
    
    print("Đang đọc dữ liệu kiểm thử...")
    df = pd.read_csv(csv_path)
    
    # Chọn 1 chuyến đi (trip_id) bất kỳ để vẽ (ở đây lấy trip cuối cùng)
    last_trip_id = df['trip_id'].unique()[-1]
    trip_df = df[df['trip_id'] == last_trip_id].reset_index(drop=True)
    
    print(f"Mô phỏng chuyến đi ID: {last_trip_id} với {len(trip_df)} khung hình.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionAutoencoder(LATENT_DIM).to(device)
    
    # Load trọng số
    model_path = r"e:\robot-jeston\experiments\weights\vision_autoencoder.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Đã load model từ '{model_path}'")
    else:
        print("Không tìm thấy model, vui lòng chạy train trước!")
        return

    model.eval()
    
    # Pre-load toàn bộ ảnh của chuyến đi vào RAM (vì RAM máy tính dư dả)
    imgs_tensor = []
    y_true = []
    
    for _, row in tqdm(trip_df.iterrows(), total=len(trip_df), desc="Loading Images"):
        img_path = os.path.join(images_dir, row['image_path'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Ảnh lỗi thì lấp ảnh đen
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) # (1, H, W)
        imgs_tensor.append(img)
        
        y_true.append([row[TARGETS[0]], row[TARGETS[1]]])
        
    X_tensor = torch.tensor(np.stack(imgs_tensor, axis=0)) # (N, C, H, W)
    y_true = np.array(y_true)
    
    print("Tiến hành suy luận (Inference)...")
    y_pred_list = []
    recon_list = []
    
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), batch_size)):
            batch_X = X_tensor[i : i + batch_size].to(device)
            # Quan trọng: Đặt mode='predictor' để AI xuất ra góc lái thay vì xuất ra ảnh
            preds = model(batch_X, mode='predictor')
            y_pred_list.append(preds.cpu().numpy())
            
    y_pred = np.vstack(y_pred_list)
    
    # Trích xuất 10 khung hình ngẫu nhiên để xem model tái tạo ảnh (Autoencoder) như thế nào
    print("Tái tạo ảnh ngẫu nhiên...")
    indices = np.random.choice(len(X_tensor), size=10, replace=False)
    sample_X = X_tensor[indices].to(device)
    with torch.no_grad():
        recon_X = model(sample_X, mode='autoencoder').cpu().numpy()
    sample_X = sample_X.cpu().numpy()
    
    # ==========================================
    # VẼ BIỂU ĐỒ 
    # ==========================================
    print("Đang vẽ biểu đồ...")
    plt.figure(figsize=(15, 8))
    
    # 1. Góc lái (Steer)
    plt.subplot(2, 1, 1)
    plt.plot(y_true[:, 0], label='Thực tế (Human)', color='blue', alpha=0.7, linestyle='dashed')
    plt.plot(y_pred[:, 0], label='Dự đoán (AI Vision)', color='red', alpha=0.9)
    plt.title(f'Góc Lái (Steer) - Chuyến đi {last_trip_id}')
    plt.xlabel('Khung hình (thời gian)')
    plt.ylabel('Giá trị bẻ lái')
    plt.legend()
    plt.grid(True)
    
    # 2. Tốc độ (Throttle)
    plt.subplot(2, 1, 2)
    plt.plot(y_true[:, 1], label='Thực tế (Human)', color='green', alpha=0.7, linestyle='dashed')
    plt.plot(y_pred[:, 1], label='Dự đoán (AI Vision)', color='orange', alpha=0.9)
    plt.title(f'Vận Tốc (Throttle) - Chuyến đi {last_trip_id}')
    plt.xlabel('Khung hình (thời gian)')
    plt.ylabel('Giá trị ga')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(r'e:\robot-jeston\experiments\plots\vision_continuous_visualization.png')
    print("-> Đã lưu biểu đồ steer/throttle tại 'plots/vision_continuous_visualization.png'")
    
    # Vẽ ảnh tái tạo
    num_samples = len(indices)
    plt.figure(figsize=(10, 2 * num_samples))
    for i in range(num_samples):
        orig_img = sample_X[i] # (1, H, W)
        recon_img = recon_X[i]
        
        # Bỏ đi channel dimension để matplotlib hiển thị ảnh xám
        orig_img = orig_img.squeeze(0)
        recon_img = recon_img.squeeze(0)
        
        # Bức ảnh gốc (bên trái)
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(orig_img, cmap='gray')
        plt.title(f"Ảnh Gốc (ROI) - Frame {indices[i]}")
        plt.axis('off')
        
        # Bức ảnh AI tái tạo (bên phải)
        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(recon_img, cmap='gray')
        plt.title("AI Tái tạo (Reconstructed)")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(r'e:\robot-jeston\experiments\plots\vision_autoencoder_reconstruction.png')
    print("-> Đã lưu biểu đồ tái tạo ảnh tại 'plots/vision_autoencoder_reconstruction.png'")
    
if __name__ == "__main__":
    test_continuous()
