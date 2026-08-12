import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

TARGETS = ['steer_raw', 'target_speed']
IMG_WIDTH = 128
IMG_HEIGHT = 32
LATENT_DIM = 128

class VisionDataset(Dataset):
    def __init__(self, dataset_dir):
        self.images_dir = os.path.join(dataset_dir, "images")
        csv_path = os.path.join(dataset_dir, "dataset_labels.csv")
        
        df = pd.read_csv(csv_path)
        self.samples = []
        
        for _, row in df.iterrows():
            target = np.array([row[TARGETS[0]], row[TARGETS[1]]], dtype=np.float32)
            self.samples.append({
                'img_path': row['image_path'],
                'target': target
            })
                
        print(f"Tổng cộng {len(self.samples)} khung hình JPG.")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.images_dir, sample['img_path'])
        
        # Đọc ảnh xám (1 channel)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) # C, H, W (1, H, W)
        
        return torch.tensor(img), torch.tensor(sample['target'])

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            # Input: (1, 32, 128)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # (32, 16, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # (64, 8, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Output: (128, 4, 16)
            nn.Flatten(),
            nn.Linear(128 * 4 * 16, latent_dim)
        )
    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 16),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            # (128, 4, 16) -> (64, 8, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # (64, 8, 32) -> (32, 16, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # (32, 16, 64) -> (1, 32, 128)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 4, 16)
        return self.deconv(x)

class ControlPredictor(nn.Module):
    def __init__(self, latent_dim=128):
        super(ControlPredictor, self).__init__()
        
        # Nhánh chuyên biệt học góc lái (Steer)
        self.steer_branch = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Nhánh chuyên biệt học vận tốc (Throttle)
        self.throttle_branch = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        steer = self.steer_branch(x)
        throttle = self.throttle_branch(x)
        # Gộp lại thành tensor 1D (Batch, 2) để tương thích với các script train/test cũ
        return torch.cat((steer, throttle), dim=1)

class VisionAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(VisionAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.predictor = ControlPredictor(latent_dim)

    def forward(self, x, mode='autoencoder'):
        # x có shape (Batch, C, H, W)
        z = self.encoder(x)
        
        if mode == 'autoencoder':
            return self.decoder(z)
        elif mode == 'predictor':
            return self.predictor(z)


