# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:12:45 2026

@author: wding64
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ==========================================
# 1. 配置与参数
# ==========================================
# 必须与训练时的路径保持一致
ROOT_DIR = "./"
DATA_DIR = os.path.join(ROOT_DIR, "NNLCI_Data/Config3_Stencils/")
MODEL_DIR = os.path.join(ROOT_DIR, "NNLCI_Models/")
OUTPUT_DIR = os.path.join(ROOT_DIR, "NNLCI_Output/")
CONFIG_ID = "3"

# [关键] 这里需要你填入 Config3 对应的网格尺寸，用于计算 SSIM
# 如果不知道，可以设为 None，脚本会自动跳过 SSIM 计算
# 例如：如果网格是 256x256，则填 (256, 256)
GRID_SIZE = None  # 格式: (Height, Width) 例如 (400, 400) 或 (256, 256)

BATCH_SIZE = 65536
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型权重文件路径 (确保文件名与训练保存的一致)
MODEL_PATH = os.path.join(MODEL_DIR, f"nnlci_config{CONFIG_ID}_deep_latest.pth")

# ==========================================
# 2. 类定义 (必须与训练代码完全一致)
# ==========================================
class NNLCIDataset(Dataset):
    def __init__(self, input_path, target_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        self.input_data = np.load(input_path, mmap_mode='r')
        self.target_data = np.load(target_path, mmap_mode='r')
        self.length = self.input_data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.from_numpy(self.input_data[idx].copy()).float()
        y = torch.from_numpy(self.target_data[idx].copy()).float()
        return x, y

class NNLCI_Net(nn.Module):
    def __init__(self, input_dim=72, output_dim=4, hidden_layers=None):
        super(NNLCI_Net, self).__init__()
        if hidden_layers is None:
            hidden_layers = [600] * 10 
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh()) 
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 评估逻辑
# ==========================================
def evaluate():
    print(f">>> Loading Model from {MODEL_PATH}...")
    # 初始化模型结构
    deep_layers = [600] * 10
    model = NNLCI_Net(input_dim=72, output_dim=4, hidden_layers=deep_layers).to(DEVICE)
    
    # 加载权重
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # 加载数据 (这里我们加载完整的训练数据作为演示，如果有单独的 test 文件请修改文件名)
    # 注意：shuffle=False 是必须的，为了后续可能的图像重构
    test_input_file = os.path.join(DATA_DIR, f"train_input_config{CONFIG_ID}.npy")
    test_target_file = os.path.join(DATA_DIR, f"train_target_config{CONFIG_ID}.npy")
    
    print(f">>> Loading Dataset: {test_input_file}")
    dataset = NNLCIDataset(test_input_file, test_target_file)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # 存储预测结果和真实值
    all_preds = []
    all_targets = []

    print(">>> Starting Inference...")
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            pred = model(x)
            
            # 将结果转回 CPU 并存入列表
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    # 合并所有 batch
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"Prediction Shape: {all_preds.shape}")

    # ==========================================
    # 4. 计算指标
    # ==========================================
    print("\n>>> Calculating Metrics...")

    # 1. L1 Loss (MAE)
    l1_loss = np.mean(np.abs(all_preds - all_targets))
    
    # 2. L2 Loss (MSE)
    l2_loss = np.mean((all_preds - all_targets) ** 2)
    
    # 3. Relative L1 & L2 (使用范数计算)
    # 防止分母为0，加一个小 epsilon
    epsilon = 1e-8
    
    # 计算整体的 Relative Error (Global)
    # Norm 2
    norm_diff_l2 = np.linalg.norm(all_preds - all_targets)
    norm_target_l2 = np.linalg.norm(all_targets)
    rel_l2 = norm_diff_l2 / (norm_target_l2 + epsilon)

    # Norm 1
    norm_diff_l1 = np.sum(np.abs(all_preds - all_targets))
    norm_target_l1 = np.sum(np.abs(all_targets))
    rel_l1 = norm_diff_l1 / (norm_target_l1 + epsilon)

    print("-" * 30)
    print(f"L1 Loss (MAE)      : {l1_loss:.6e}")
    print(f"L2 Loss (MSE)      : {l2_loss:.6e}")
    print(f"Relative L1 Error  : {rel_l1:.6e}")
    print(f"Relative L2 Error  : {rel_l2:.6e}")
    print("-" * 30)

    # ==========================================
    # 5. PSNR 和 SSIM
    # ==========================================
    # 数据范围 (Data Range) 对于 PSNR 至关重要
    # NNLCI 输出通常是权重，范围可能在 [0, 1] 或 [-1, 1] 之间，我们从数据中动态获取
    data_range = all_targets.max() - all_targets.min()
    
    # 计算 PSNR (Flattened data is fine for generic PSNR, though technically it's an image metric)
    # skimage 的 psnr 既支持图像也支持数组
    psnr_val = psnr(all_targets, all_preds, data_range=data_range)
    print(f"PSNR               : {psnr_val:.4f} dB")

    # 计算 SSIM (需要 Reshape)
    if GRID_SIZE is not None:
        H, W = GRID_SIZE
        total_pixels = H * W
        
        # 检查数据量是否匹配
        if all_preds.shape[0] == total_pixels:
            print(f"\n>>> Reshaping to image ({H}, {W}) for SSIM...")
            
            # Reshape: (N, 4) -> (H, W, 4)
            # 假设数据是按行或列优先存储的连续网格数据
            img_pred = all_preds.reshape(H, W, -1)
            img_target = all_targets.reshape(H, W, -1)
            
            # SSIM 需要计算 multi-channel
            # win_size 必须小于图像的最小边长，默认为7，如果是小网格需要调整
            try:
                ssim_val = ssim(img_target, img_pred, data_range=data_range, channel_axis=-1)
                print(f"SSIM               : {ssim_val:.6f}")
            except ValueError as e:
                print(f"SSIM Calculation Error: {e} (Check window size or image dims)")
        else:
            print(f"\n[Warning] Data length {all_preds.shape[0]} does not match Grid Size {H}x{W}={total_pixels}.")
            print("Skipping SSIM calculation.")
    else:
        print("\n[Info] GRID_SIZE is None. Skipping SSIM calculation.")
        print("To compute SSIM, please set GRID_SIZE = (Height, Width) at the top of the script.")

    # ==========================================
    # 6. 保存预测结果 (可选)
    # ==========================================
    save_path = os.path.join(OUTPUT_DIR, f"predictions_config{CONFIG_ID}.npy")
    np.save(save_path, all_preds)
    print(f"\nPredictions saved to: {save_path}")

if __name__ == "__main__":
    evaluate()