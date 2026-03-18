# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 19:38:37 2026

@author: wding64
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view

# --- 配置 ---
NUM_WORKERS = 0  # 调试时建议设为 0，避免多进程报错掩盖真实错误
BATCH_SIZE = 65536 
USE_AMP = False  # 🔥 调试建议：先关闭 AMP，排查数值问题
PIN_MEMORY = True 

ROOT_DIR = "./" 
MODEL_FILENAME = 'best_model_800.pth'
WEIGHTS_PATH = os.path.join(ROOT_DIR, 'WC_NNLCI', MODEL_FILENAME)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Training for 800-Grid Super-Res on: {torch.cuda.get_device_name(0)}")

# 调试环境变量 (可选，如果报错能看到更准的位置)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# --- 数据处理核心 ---
def load_and_process_data(root_dir):
    print("📦 Loading raw data...")
    data_dir = os.path.join(root_dir, 'WC_NNLCI')
    
    # 1. 加载原始数据
    train_100 = np.load(os.path.join(data_dir, 'training_dataset_100.npy'))
    train_200 = np.load(os.path.join(data_dir, 'training_dataset_200.npy'))
    train_1600 = np.load(os.path.join(data_dir, 'training_dataset_hf.npy')).reshape((401, 1600, 3))

    # 选取训练集 Case
    indices = np.arange(0, 400, 10) 
    train_100_src = train_100[indices]     # [40, 100, 3]
    train_200_src = train_200[indices]     # [40, 200, 3]
    
    # 构造 800 Grid Target
    train_800_target = train_1600[indices, ::2, :] # [40, 800, 3]
    
    print("⚡ Interpolating Inputs (100/200 -> 800)...")
    x_100 = np.linspace(0, 1, 100)
    x_200 = np.linspace(0, 1, 200)
    x_800 = np.linspace(0, 1, 800)
    
    input_100_interp = np.zeros_like(train_800_target)
    input_200_interp = np.zeros_like(train_800_target)
    
    for i in range(len(indices)):
        for v in range(3):
            f1 = interp1d(x_100, train_100_src[i, :, v], kind='linear', fill_value="extrapolate")
            input_100_interp[i, :, v] = f1(x_800)
            f2 = interp1d(x_200, train_200_src[i, :, v], kind='linear', fill_value="extrapolate")
            input_200_interp[i, :, v] = f2(x_800)

    # --- 🔥 安全性检查 (Fixing the Bug) ---
    # 检查最小值，确保 log10(x+2) 安全
    min_val = min(train_800_target.min(), input_100_interp.min(), input_200_interp.min())
    print(f"🧐 Data Min Value: {min_val}")
    
    if min_val <= -2.0:
        print("⚠️ Warning: Data contains values <= -2. Adjusting shift...")
        shift_val = abs(min_val) + 1.0 # 动态调整 shift
    else:
        shift_val = 2.0
        
    print(f"🔧 Using Log Shift: {shift_val}")

    def transform(x): 
        # 添加极小量 eps 防止刚好为0
        return np.log10(x + shift_val + 1e-6) 
    
    train_800_target = transform(train_800_target)
    input_100_interp = transform(input_100_interp)
    input_200_interp = transform(input_200_interp)
    
    # 再次检查 NaN
    if np.isnan(train_800_target).any() or np.isnan(input_100_interp).any():
        raise ValueError("❌ NaN detected after transformation! Check your data.")

    # 计算 Scaling
    ub = np.max(input_100_interp, axis=(0, 1))
    lb = np.min(input_100_interp, axis=(0, 1))
    
    # 防止除以 0
    diff = ub - lb
    diff[diff == 0] = 1.0
    
    def scale(x, lb, ub): return 2 * (x - lb) / diff - 1
    
    target_norm = scale(train_800_target, lb, ub)
    in100_norm  = scale(input_100_interp, lb, ub)
    in200_norm  = scale(input_200_interp, lb, ub)
    
    # --- 构建 Dataset (优化版) ---
    print("✂️ Slicing windows efficiently...")
    
    # 目标：Input Window [i, i+1, i+2] -> Predict Target [i+1]
    # 我们有 800 个点。
    # 有效中心点是 index 1 到 798 (共 798 个点)
    # 对应的 Window start index 是 0 到 797
    
    # 使用 sliding_window_view 替代慢循环
    # Shape: [40, 800, 3] -> Window axis=1, size=3
    # Result: [40, 798, 3, 3] (798 windows: 0..2, 1..3, ... 797..799)
    
    win_100 = sliding_window_view(in100_norm, window_shape=3, axis=1) # [40, 798, 3, 3]
    win_200 = sliding_window_view(in200_norm, window_shape=3, axis=1) # [40, 798, 3, 3]
    
    # Target 对应的中心点是 index 1 到 798
    # Slice: [:, 1:-1, :] -> 从 index 1 开始，到倒数第1个之前
    target_slice = target_norm[:, 1:-1, :] # [40, 798, 3]
    
    # 调整维度以符合你的 concat 逻辑
    # win_100: [40, 798, 3, 3] -> flatten last two -> [40, 798, 9]
    X_c = win_100.reshape(40, 798, 9)
    X_f = win_200.reshape(40, 798, 9)
    Y   = target_slice # [40, 798, 3]
    
    # 合并 Flatten
    X_train = np.concatenate([X_c, X_f], axis=2).reshape(-1, 18) # [31920, 18]
    Y_train = Y.reshape(-1, 3)
    
    print(f"✅ Final Dataset Shape: X={X_train.shape}, Y={Y_train.shape}")
    return torch.FloatTensor(X_train), torch.FloatTensor(Y_train), lb, ub

# --- Fully Connected Model (保持不变) ---
class NeuralNet(nn.Module):
    def __init__(self, layers):
        super(NeuralNet, self).__init__()
        self.depth = len(layers) - 1
        self.actfunc = nn.Tanh()
        self.linears = nn.ModuleList()
        for i in range(self.depth):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(layer.weight.data, gain=5.0/3.0)
            nn.init.zeros_(layer.bias.data)
            self.linears.append(layer)

    def forward(self, x):
        for i in range(self.depth - 1):
            x = self.actfunc(self.linears[i](x))
        x = self.linears[-1](x)
        return x

# --- 主程序 ---
if __name__ == '__main__':
    # 强制让 PyTorch 报错更具体 (CPU 检查)
    torch.autograd.set_detect_anomaly(True)
    
    X_train, Y_train, lb, ub = load_and_process_data(ROOT_DIR)
    
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,    
        pin_memory=PIN_MEMORY
    )

    layers = [18] + 10*[300] + [3]
    model = NeuralNet(layers).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500)
    
    # 如果开启 AMP
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    criterion = nn.MSELoss()

    print("🚀 Starting Training...")
    model.train()
    
    EPOCHS = 10000 
    
    for epoch in tqdm(range(EPOCHS)):
        epoch_loss = 0.0
        batches = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 使用 AMP 上下文
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                preds = model(inputs)
                loss = criterion(preds, targets)
            
            if torch.isnan(loss):
                print(f"❌ Loss is NaN at Epoch {epoch}! Stopping.")
                exit()

            scaler.scale(loss).backward()
            
            # 增加梯度裁剪，防止梯度爆炸触发 CUDA Error
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            batches += 1
        
        avg_loss = epoch_loss / batches if batches > 0 else 0
        scheduler.step(avg_loss)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} Loss: {avg_loss:.4e} LR: {optimizer.param_groups[0]['lr']:.1e}")
            torch.save(model.state_dict(), WEIGHTS_PATH)
            
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print("✅ Done!")