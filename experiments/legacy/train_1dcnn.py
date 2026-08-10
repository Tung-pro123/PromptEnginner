import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. Cấu hình Tham số (Hyperparameters)
# ==========================================
K_STEPS = 10         # Chiều dài cửa sổ (K bước quá khứ)
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4

# Các đặc trưng (features) dùng làm input
FEATURES = [
    'lateral_error_m', 
    'heading_error_deg', 
    'curvature', 
    'actual_speed',
    'steer_raw'
]

# Các giá trị mục tiêu (targets)
# Chọn steer_filtered thay vì steer_raw để bẻ lái mượt mà
TARGETS = ['steer_filtered', 'target_speed']

# ==========================================
# 2. Xử lý dữ liệu & Sliding Window
# ==========================================
def load_and_preprocess_data(logs_dir):
    csv_files = glob.glob(os.path.join(logs_dir, "*.csv"))
    
    all_features = []
    all_targets = []
    
    for f in csv_files:
        df = pd.read_csv(f)
        
        # Bỏ qua các hàng bị lỗi hoặc thiếu dữ liệu quan trọng
        df = df.dropna(subset=FEATURES + TARGETS)
        
        if len(df) < K_STEPS:
            continue
            
        # TÍNH NĂNG MỚI: Vuốt mượt nhãn (Label Smoothing)
        # Dùng Moving Average (trung bình trượt) với cửa sổ 5 bước để khử các đỉnh giật cục 
        # của vô lăng trong file log gốc. AI sẽ học cách lái theo đường cong mượt này.
        for target_col in TARGETS:
            df[target_col] = df[target_col].rolling(window=7, min_periods=1, center=True).mean()
            
        all_features.append(df[FEATURES].values)
        all_targets.append(df[TARGETS].values)
        
    if not all_features:
        raise ValueError("Không tìm thấy dữ liệu hợp lệ trong các file CSV.")
        
    # Gộp tất cả các file lại (lưu ý: cách này đơn giản hóa, thực tế nên trượt cửa sổ trong từng file trước)
    return all_features, all_targets

def create_sliding_windows(features_list, targets_list, k):
    X, y = [], []
    for feat, targ in zip(features_list, targets_list):
        # Trượt cửa sổ cho từng file/chuyến đi độc lập để tránh nhiễu ở điểm nối
        for i in range(len(feat) - k):
            X.append(feat[i : i + k])
            y.append(targ[i + k]) # Dự đoán hành động ở bước hiện tại k (dựa vào 0 -> k-1)
    return np.array(X), np.array(y)

class DrivingDataset(Dataset):
    def __init__(self, X, y):
        # PyTorch Conv1D expect input shape: (Batch, Channels, Length)
        # Nghĩa là: (Batch, Num_Features, K_Steps)
        self.X = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 3. Kiến trúc 1D-CNN (Phương án A)
# ==========================================
class SmoothDriveCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmoothDriveCNN, self).__init__()
        
        # Các lớp Convolution 1D dọc theo thời gian
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Flatten và Dense layers
        # Với K=10, MaxPool(2) sẽ giảm chiều dài xuống 10/2 = 5
        self.fc_blocks = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (K_STEPS // 2), 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Ngăn chặn Overfitting
            nn.Linear(128, out_channels)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc_blocks(x)
        return x

# ==========================================
# 4. Hàm Loss chống giật (Smooth L1 Loss)
# ==========================================
def train_model():
    logs_dir = r"e:\robot-jeston\logs\logs"
    print("Đang đọc và tiền xử lý dữ liệu...")
    
    features_list, targets_list = load_and_preprocess_data(logs_dir)
    
    # Gom tất cả feature lại để fit StandardScaler (tránh data leakage)
    all_feat_concat = np.vstack(features_list)
    scaler = StandardScaler()
    scaler.fit(all_feat_concat)
    
    # Normalize từng file
    features_list_scaled = [scaler.transform(f) for f in features_list]
    
    # Tạo Sliding windows
    X, y = create_sliding_windows(features_list_scaled, targets_list, K_STEPS)
    
    # Chia tập Train/Test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    train_dataset = DrivingDataset(X_train, y_train)
    test_dataset = DrivingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Khởi tạo Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmoothDriveCNN(in_channels=len(FEATURES), out_channels=len(TARGETS)).to(device)
    
    # Sử dụng SmoothL1Loss (Huber Loss) thay cho MSE để ít nhạy cảm với các pha bẻ lái nhiễu (outliers) -> Giúp lái mượt hơn
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Bắt đầu huấn luyện trên {device} với {len(X_train)} mẫu train và {len(X_test)} mẫu test...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Tính loss
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(test_loader.dataset)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss (Huber): {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    print("Hoàn tất huấn luyện!")
    # Lưu model và scaler (cần scaler để inference sau này)
    torch.save(model.state_dict(), "smooth_drive_cnn.pth")
    print("Đã lưu model tại 'smooth_drive_cnn.pth'")
    
    # ==========================================
    # 5. Visualization (Trực quan hóa dự đoán)
    # ==========================================
    print("Đang tạo biểu đồ trực quan hóa...")
    model.eval()
    all_preds = []
    all_trues = []
    
    # Lấy khoảng 200 mẫu của tập test để vẽ (tránh quá nhiều sẽ bị rối)
    viz_samples = 200
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).cpu().numpy()
            trues = batch_y.numpy()
            
            all_preds.append(preds)
            all_trues.append(trues)
            
            if sum(len(b) for b in all_preds) >= viz_samples:
                break
                
    all_preds = np.vstack(all_preds)[:viz_samples]
    all_trues = np.vstack(all_trues)[:viz_samples]
    
    plt.figure(figsize=(15, 5 * len(TARGETS)))
    for i, target_name in enumerate(TARGETS):
        plt.subplot(len(TARGETS), 1, i + 1)
        plt.plot(all_trues[:, i], label='Thực tế (True)', color='blue', alpha=0.7)
        plt.plot(all_preds[:, i], label='AI Dự đoán (Predicted)', color='red', linestyle='--', alpha=0.9)
        plt.title(f'So sánh Dự đoán vs Thực tế - Thuộc tính: {target_name}')
        plt.xlabel('Samples (Time steps)')
        plt.ylabel('Giá trị')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    viz_path = "prediction_visualization.png"
    plt.savefig(viz_path)
    print(f"Đã lưu biểu đồ tại '{viz_path}'")

if __name__ == "__main__":
    train_model()
