import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from train_vision_autoencoder import VisionDataset, VisionAutoencoder, LATENT_DIM

def train_stage2():
    dataset_dir = r"e:\robot-jeston\logs\dataset"
    print("Đang load dữ liệu...")
    dataset = VisionDataset(dataset_dir)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    model = VisionAutoencoder(LATENT_DIM).to(device)
    
    # Load pretrained Stage 1
    ae_path = r"e:\robot-jeston\experiments\weights\autoencoder_only.pth"
    if os.path.exists(ae_path):
        model.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True), strict=False)
        print(f"Đã load trọng số Autoencoder từ {ae_path}")
    else:
        print("CẢNH BÁO: Không tìm thấy trọng số Stage 1, mô hình sẽ train từ đầu!")

    print("\n--- [GIAI ĐOẠN 2] Đóng băng Encoder & Huấn luyện Control Predictor ---")
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    optimizer_pred = optim.Adam(model.predictor.parameters(), lr=1e-3)
    criterion_pred = nn.SmoothL1Loss()
    
    epochs_stage2 = 150
    for epoch in range(1, epochs_stage2 + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Pred Epoch {epoch}")
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer_pred.zero_grad()
            preds = model(imgs, mode='predictor')
            loss = criterion_pred(preds, targets)
            loss.backward()
            optimizer_pred.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Predictor Epoch {epoch}/{epochs_stage2} - Lỗi bẻ lái: {total_loss/len(train_loader):.4f}")
        
    save_path = r"e:\robot-jeston\experiments\weights\vision_autoencoder.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 Hoàn thành! Đã lưu mô hình (cả AE và Predictor) tại '{save_path}'")

if __name__ == "__main__":
    train_stage2()
