import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from train_vision_autoencoder import VisionDataset, VisionAutoencoder, LATENT_DIM

def train_stage1():
    dataset_dir = r"e:\robot-jeston\logs\dataset"
    print("Đang load dữ liệu...")
    dataset = VisionDataset(dataset_dir)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    model = VisionAutoencoder(LATENT_DIM).to(device)
    
    print("\n--- [GIAI ĐOẠN 1] Huấn luyện Autoencoder ---")
    optimizer_ae = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-3)
    
    # Tạo Spatial Weight Mask (Mặt nạ trọng số không gian)
    # Ảnh có kích thước H=32, W=128
    import numpy as np
    H, W = 32, 128
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)
    
    # Tạo phân bố Gaussian tập trung vào giữa theo trục X (Center_x = 64)
    # Và tập trung vào phần nửa dưới của ảnh theo trục Y (Center_y = 22) - nơi có vạch kẻ đường
    sigma_x = W / 4.0
    sigma_y = H / 2.0
    weight_x = np.exp(-((xx - W/2)**2) / (2 * sigma_x**2))
    weight_y = np.exp(-((yy - H*0.7)**2) / (2 * sigma_y**2))
    
    # Vùng trung tâm sẽ có trọng số cao gấp 5 lần (1.0 cơ bản + 4.0 vùng tâm)
    spatial_weight = 1.0 + 4.0 * (weight_x * weight_y)
    mask_tensor = torch.tensor(spatial_weight, dtype=torch.float32).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, 32, 128)
    
    epochs_stage1 = 150
    for epoch in range(1, epochs_stage1 + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"AE Epoch {epoch}")
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            
            optimizer_ae.zero_grad()
            recon = model(imgs, mode='autoencoder')
            
            # Tính MSE Loss nhân với mặt nạ không gian (tập trung tái tạo vùng trung tâm)
            squared_diff = (recon - imgs) ** 2
            weighted_squared_diff = squared_diff * mask_tensor
            loss = weighted_squared_diff.mean()
            
            loss.backward()
            optimizer_ae.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"AE Epoch {epoch}/{epochs_stage1} - Lỗi tái tạo: {total_loss/len(train_loader):.4f}")
        
    save_path = r"e:\robot-jeston\experiments\weights\autoencoder_only.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 Đã lưu trọng số Stage 1 tại '{save_path}'")

if __name__ == "__main__":
    train_stage1()
