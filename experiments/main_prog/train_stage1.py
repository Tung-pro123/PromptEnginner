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
    criterion_ae = nn.MSELoss()
    
    epochs_stage1 = 1000
    for epoch in range(1, epochs_stage1 + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"AE Epoch {epoch}")
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            
            optimizer_ae.zero_grad()
            recon = model(imgs, mode='autoencoder')
            loss = criterion_ae(recon, imgs)
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
