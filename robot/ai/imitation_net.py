import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset

class SmartCityImitationNet(nn.Module):
    """
    Mạng Neural (MLP) cho học bắt chước bài toán Smart City.
    Đầu vào: Vector Trạng thái (lane_offset, lane_curvature, sign_class, lidar_dist_front, lidar_dist_side)
    Đầu ra: Góc lái (steer) và Chân ga (throttle)
    """
    def __init__(self, input_dim=5, output_dim=2):
        super(SmartCityImitationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Tanh() # Đảm bảo output nằm trong khoảng [-1, 1] cho steer/throttle
        )
        
    def forward(self, x):
        return self.net(x)

class ImitationDataset(Dataset):
    """
    Load dữ liệu từ file CSV do Tay cầm (Joy) sinh ra.
    """
    def __init__(self, csv_files):
        # Đọc và gộp toàn bộ các file CSV lấy mẫu nháp
        dfs = [pd.read_csv(f) for f in csv_files]
        self.data = pd.concat(dfs, ignore_index=True)
        
        # X (Features): 5 tham số đầu vào
        self.features = self.data[['lane_offset', 'lane_curvature', 'sign_class', 'lidar_dist_front', 'lidar_dist_side']].values
        
        # Y (Labels): 2 tham số đầu ra (Steer, Throttle)
        self.labels = self.data[['action_steer', 'action_throttle']].values
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
