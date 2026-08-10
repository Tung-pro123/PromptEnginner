import os
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import các thành phần từ file train
from train_1dcnn import SmoothDriveCNN, FEATURES, TARGETS, K_STEPS, load_and_preprocess_data, create_sliding_windows

def test_continuous():
    logs_dir = r"e:\robot-jeston\logs\logs"
    
    # 1. Load và tiền xử lý toàn bộ dữ liệu như lúc train để tái tạo lại StandardScaler chuẩn
    # (Cách tốt nhất thực tế là lưu scaler ra file .pkl, nhưng tạm thời fit lại ở đây)
    print("Đang load lại dữ liệu để lấy hệ số chuẩn hóa (Scaler)...")
    features_list, _ = load_and_preprocess_data(logs_dir)
    all_feat_concat = np.vstack(features_list)
    scaler = StandardScaler()
    scaler.fit(all_feat_concat)
    
    # 2. Chọn 1 file CSV cụ thể để test (ví dụ file lớn nhất hoặc mới nhất)
    csv_files = glob.glob(os.path.join(logs_dir, "*.csv"))
    # Sắp xếp để lấy file mới nhất hoặc tự định nghĩa
    csv_file = sorted(csv_files)[-1] 
    
    print(f"\nĐang test liên tục (continuous) trên file: {os.path.basename(csv_file)}")
    
    # Xử lý riêng biệt cho 1 file này
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=FEATURES + TARGETS)
    
    # Tái tạo lại bước Label Smoothing cho file test để so sánh cho công bằng
    for target_col in TARGETS:
        df[target_col] = df[target_col].rolling(window=7, min_periods=1, center=True).mean()
        
    feat_1file = df[FEATURES].values
    targ_1file = df[TARGETS].values
    
    # Chuẩn hóa feature
    feat_1file_scaled = scaler.transform(feat_1file)
    
    # Tạo sliding windows (Mảng liên tục, KHÔNG SHUFFLE)
    X, y_true = create_sliding_windows([feat_1file_scaled], [targ_1file], K_STEPS)
    
    # Convert sang PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
    
    # 3. Khởi tạo và Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmoothDriveCNN(in_channels=len(FEATURES), out_channels=len(TARGETS)).to(device)
    
    model_path = "smooth_drive_cnn.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("✅ Đã load thành công trọng số từ 'smooth_drive_cnn.pth'.")
    else:
        print("⚠️ CẢNH BÁO: Không tìm thấy model đã train. AI sẽ dự đoán ngẫu nhiên!")
        
    model.eval()
    
    # 4. Chạy dự đoán toàn bộ chuỗi
    print("Đang chạy inference qua mạng AI...")
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        y_pred = model(X_tensor).cpu().numpy()
        
    # 5. Vẽ biểu đồ Visualization (Dạng chuỗi thời gian liên tục)
    print("Đang vẽ biểu đồ...")
    plt.figure(figsize=(18, 5 * len(TARGETS)))
    
    for i, target_name in enumerate(TARGETS):
        plt.subplot(len(TARGETS), 1, i + 1)
        
        # Đường thực tế
        plt.plot(y_true[:, i], label='Thực tế (Smoothed True)', color='royalblue', linewidth=2, alpha=0.8)
        
        # Đường dự đoán
        plt.plot(y_pred[:, i], label='AI Dự đoán (Predicted)', color='crimson', linestyle='--', linewidth=2, alpha=0.9)
        
        plt.title(f'Test Liên tục - File: {os.path.basename(csv_file)} | Thuộc tính: {target_name}', fontsize=14)
        plt.xlabel('Khung hình (Time steps liên tục)', fontsize=12)
        plt.ylabel('Giá trị', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        
    plt.tight_layout()
    viz_path = "continuous_visualization.png"
    plt.savefig(viz_path, dpi=150)
    print(f"🎉 Đã lưu biểu đồ chuỗi thời gian liên tục tại: '{viz_path}'")

if __name__ == "__main__":
    test_continuous()
