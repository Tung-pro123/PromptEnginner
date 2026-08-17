#!/usr/bin/env python3
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from robot.ai.imitation_net import SmartCityImitationNet, ImitationDataset

def train_and_test():
    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Tìm toàn bộ dữ liệu CSV (Các lần đào tạo nháp)
    csv_files = glob.glob(os.path.join(log_dir, '*.csv'))
    if not csv_files:
        print("[-] Không tìm thấy dữ liệu mẫu (CSV) nào trong thư mục logs/. Vui lòng dùng tay cầm chạy để lấy mẫu trước!")
        return
        
    print(f"[+] Đang tải {len(csv_files)} file dữ liệu lấy mẫu...")
    dataset = ImitationDataset(csv_files)
    print(f"[+] Tổng số mẫu (frames): {len(dataset)}")
    
    # 2. Chia tập Train / Test (80% Train, 20% Validate)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Khởi tạo Mô hình
    model = SmartCityImitationNet(input_dim=5, output_dim=2)
    criterion = nn.MSELoss() # Dùng sai số toàn phương trung bình cho bài toán hồi quy (Steer, Throttle)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    print("[+] Bắt đầu quá trình Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Kiểm tra hiệu suất trên tập Test (Validate) mỗi 10 epoch
        if (epoch+1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, y_val in test_loader:
                    pred_val = model(x_val)
                    val_loss += criterion(pred_val, y_val).item()
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Test/Val Loss: {val_loss/len(test_loader):.4f}")
            
    # 4. Lưu mô hình (Trọng số)
    model_path = os.path.join(model_dir, 'imitation_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"[+] Đào tạo hoàn tất! Đã lưu mô hình tại: {model_path}")

if __name__ == "__main__":
    train_and_test()
