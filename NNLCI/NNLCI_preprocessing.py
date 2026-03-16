# -*- coding: utf-8 -*-
"""
NNLCI 2.0 Preprocessor with Shock Masking (Smart Sampling)
- Integrated into the corrected base file.
- Features:
  1. Multi-Fidelity Input (100x & 200x -> 800x)
  2. Smart Sampling: Keeps 40% Shocks + 10% Smooth regions.
  3. Robust Debugging: Prints file paths if loading fails.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import re
import io
import random
from tqdm import tqdm

# ==========================================
# 1. 配置参数
# ==========================================
ROOT_DIR = "./"
DATA_ROOT = os.path.join(ROOT_DIR, "DATA_TRAIN")

CONFIG_ID = "3" 
OUTPUT_DIR = os.path.join(ROOT_DIR, "NNLCI_Data", f"Config{CONFIG_ID}_Stencils")

# 采样设置
TOTAL_TRAIN_SAMPLES = 200
SAMPLES_TO_PICK = 20
RANDOM_SEED = 42

# [新增] Smart Sampling 比率
SHOCK_RATIO = 0.4   # 保留梯度最大的 40% (激波区)
SMOOTH_RATIO = 0.1  # 从剩余平滑区保留 10% (背景区)

# 分辨率
RES_LF1 = 100
RES_LF2 = 200
RES_HF  = 800

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. 工具类与函数
# ==========================================
class MinMaxScalerMinus1To1:
    """归一化到 [-1, 1]"""
    def __init__(self, eps=1e-12):
        self.xmin = None; self.xmax = None; self.eps = eps
    def fit(self, x: np.ndarray):
        self.xmin = x.min(axis=0).astype(np.float64)
        self.xmax = x.max(axis=0).astype(np.float64)
        same = np.isclose(self.xmax, self.xmin)
        self.xmax[same] = self.xmin[same] + 1.0
    def transform(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.xmin) / (self.xmax - self.xmin + self.eps) - 1.0
    def save(self, path: str):
        np.savez(path, xmin=self.xmin, xmax=self.xmax, eps=self.eps)

def read_tecplot_dat(filepath, target_res):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        header_lines = 0
        for i, line in enumerate(lines):
            if "ZONE" in line:
                header_lines = i + 1; break
        
        raw_str = "".join(lines[header_lines:])
        fixed_str = re.sub(r'(\d)([+-]\d{3})', r'\1E\2', raw_str)
        
        data = np.loadtxt(io.StringIO(fixed_str))
        feat = data[:, 2:6] # X, Y, D, U, V, P
        
        side = int(np.sqrt(feat.shape[0]))
        feat = feat.reshape(side, side, 4)
        
        if side > target_res:
            feat = feat[:target_res, :target_res, :]
        elif side < target_res:
            return None # Skip low res
            
        return feat.transpose(2, 0, 1)

    except Exception as e:
        # 这里的 Error 不打印，以免刷屏，但在 Debug 模式下很有用
        return None

def pytorch_upsample(data_np, target_res):
    t_data = torch.from_numpy(data_np).float().to(DEVICE)
    t_up = F.interpolate(t_data, size=(target_res, target_res), mode='bilinear', align_corners=True)
    return t_up.cpu().numpy()

# ==========================================
# [核心新增] 3. Shock Masking 逻辑
# ==========================================
def compute_gradient_mask(data_tensor, high_ratio=0.4, low_ratio=0.1):
    """
    计算梯度遮罩，用于筛选激波区域。
    """
    B, C, H, W = data_tensor.shape
    
    # 简单的 Sobel 算子
    kx = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], device=DEVICE).view(1, 1, 3, 3)
    ky = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], device=DEVICE).view(1, 1, 3, 3)
    
    # 计算所有通道梯度的平均模长
    grad_mag_sum = torch.zeros((B, H, W), device=DEVICE)
    data_pad = F.pad(data_tensor, (1, 1, 1, 1), mode='replicate')
    
    for c in range(C):
        c_data = data_pad[:, c:c+1, :, :]
        gx = F.conv2d(c_data, kx)
        gy = F.conv2d(c_data, ky)
        grad_mag_sum += torch.sqrt(gx**2 + gy**2).squeeze(1)
        
    flat_grad = grad_mag_sum.view(-1)
    
    # 1. 激波区 (High Gradient)
    k_high = int(high_ratio * flat_grad.numel())
    _, high_indices = torch.topk(flat_grad, k_high)
    mask_flat = torch.zeros_like(flat_grad, dtype=torch.bool)
    mask_flat[high_indices] = True
    
    # 2. 平滑背景区 (Random Low Gradient)
    low_indices = torch.nonzero(~mask_flat).squeeze()
    num_low = int(low_ratio * low_indices.numel())
    if num_low > 0:
        perm = torch.randperm(low_indices.numel(), device=DEVICE)[:num_low]
        mask_flat[low_indices[perm]] = True
        
    print(f"    > Mask Logic: Top {high_ratio:.0%} Shocks + {low_ratio:.0%} Background.")
    print(f"    > Keeping {mask_flat.float().mean().item():.2%} of total data.")
    
    return mask_flat.view(B, H, W)

def extract_patches_masked(img_lf1, img_lf2, img_hf, device='cpu'):
    """
    带 Mask 的 Stencil 提取函数。
    """
    t_lf1 = torch.from_numpy(img_lf1).float().to(device)
    t_lf2 = torch.from_numpy(img_lf2).float().to(device)
    t_hf  = torch.from_numpy(img_hf).float().to(device)
    
    B, C, H, W = t_lf1.shape
    
    # 配置
    k_size = 3; dilation = 4; border = 4 
    
    # 1. 计算 Mask (基于 HF 真实值)
    print("    > Computing Shock Mask...")
    full_mask = compute_gradient_mask(t_hf, high_ratio=SHOCK_RATIO, low_ratio=SMOOTH_RATIO)
    
    # [关键] 裁剪 Mask 以匹配 Stencil 中心区域 (792x792)
    mask_cropped = full_mask[:, border:-border, border:-border] 
    mask_flat = mask_cropped.reshape(-1) # Flatten
    
    # 2. 提取 LF1 Stencils
    patches_lf1 = F.unfold(t_lf1, kernel_size=k_size, dilation=dilation)
    patches_lf1 = patches_lf1.permute(0, 2, 1).contiguous().reshape(-1, C*9)
    
    # 3. 提取 LF2 Stencils
    patches_lf2 = F.unfold(t_lf2, kernel_size=k_size, dilation=dilation)
    patches_lf2 = patches_lf2.permute(0, 2, 1).contiguous().reshape(-1, C*9)
    
    # 4. 提取 HF Targets
    hf_center = t_hf[:, :, border:-border, border:-border]
    hf_targets = hf_center.permute(0, 2, 3, 1).contiguous().reshape(-1, C)
    
    # 5. 应用 Mask 筛选
    print(f"    > Filtering Data: {mask_flat.numel()} -> {mask_flat.sum().item()} samples")
    
    patches_lf1 = patches_lf1[mask_flat]
    patches_lf2 = patches_lf2[mask_flat]
    hf_targets  = hf_targets[mask_flat]
    
    # 6. 拼接
    inputs = torch.cat([patches_lf1, patches_lf2], dim=1)
    
    return inputs.cpu().numpy(), hf_targets.cpu().numpy()

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    print(f">>> Starting Preprocessing (Masked) for Config {CONFIG_ID}...")
    
    # 路径检查
    config_path = os.path.join(DATA_ROOT, f"Config{CONFIG_ID}")
    print(f"    Search Path: {os.path.abspath(config_path)}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config folder not found: {config_path}")

    all_cases = sorted(glob.glob(os.path.join(config_path, "case_*")))
    
    if len(all_cases) == 0:
        raise ValueError(f"No cases found in {config_path}. Check your directory structure!")
    
    print(f"    Found {len(all_cases)} total cases.")

    # 随机采样
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if len(all_cases) > SAMPLES_TO_PICK:
        selected_cases = random.sample(all_cases, SAMPLES_TO_PICK)
    else:
        selected_cases = all_cases
    
    print(f"    Selected {len(selected_cases)} cases for processing.")
    
    # 加载数据 (带 Debug)
    list_lf1, list_lf2, list_hf = [], [], []
    
    print("\n>>> Loading Data...")
    for i, case_dir in enumerate(tqdm(selected_cases)):
        f100 = os.path.join(case_dir, f"flow_{RES_LF1}x{RES_LF1}.dat")
        f200 = os.path.join(case_dir, f"flow_{RES_LF2}x{RES_LF2}.dat")
        f800 = os.path.join(case_dir, f"flow_{RES_HF}x{RES_HF}.dat")
        
        # 打印第一个 Case 的路径以供检查
        if i == 0:
            print(f"\n[Debug] First Case Files:")
            print(f"  - {f100} : {'FOUND' if os.path.exists(f100) else 'MISSING'}")
            print(f"  - {f200} : {'FOUND' if os.path.exists(f200) else 'MISSING'}")
            print(f"  - {f800} : {'FOUND' if os.path.exists(f800) else 'MISSING'}")

        if not (os.path.exists(f100) and os.path.exists(f200) and os.path.exists(f800)):
            continue
            
        d100 = read_tecplot_dat(f100, RES_LF1)
        d200 = read_tecplot_dat(f200, RES_LF2)
        d800 = read_tecplot_dat(f800, RES_HF)
        
        if d100 is not None and d200 is not None and d800 is not None:
            list_lf1.append(d100)
            list_lf2.append(d200)
            list_hf.append(d800)

    # 检查是否加载成功
    if len(list_lf1) == 0:
        print("\n[CRITICAL ERROR] No data loaded! See the [Debug] info above.")
        print("Possible reasons:")
        print("1. File paths are wrong (check ROOT_DIR).")
        print("2. Filenames don't match (e.g., 'flow_100x100.dat' vs 'solution.dat').")
        exit(1)

    arr_lf1 = np.stack(list_lf1)
    arr_lf2 = np.stack(list_lf2)
    arr_hf  = np.stack(list_hf)
    
    print(f"\n    Loaded Batch Shapes: LF1={arr_lf1.shape}, HF={arr_hf.shape}")

    # Upsampling
    print("\n>>> Upsampling LF data...")
    arr_lf1_up = pytorch_upsample(arr_lf1, RES_HF)
    arr_lf2_up = pytorch_upsample(arr_lf2, RES_HF)

    # Extract with MASK
    print("\n>>> Extracting Stencils (Smart Sampling)...")
    train_X, train_Y = extract_patches_masked(arr_lf1_up, arr_lf2_up, arr_hf, device=DEVICE)
    
    print(f"    Final Dataset Shapes:")
    print(f"    Input X: {train_X.shape}") 
    print(f"    Target Y: {train_Y.shape}")

    # Scaling
    print("\n>>> Scaling and Saving...")
    scaler = MinMaxScalerMinus1To1()
    train_X = np.ascontiguousarray(train_X, dtype=np.float32)
    train_Y = np.ascontiguousarray(train_Y, dtype=np.float32)
    
    scaler.fit(train_X)
    scaler.save(os.path.join(OUTPUT_DIR, f'x_scaler_config{CONFIG_ID}.npz'))
    
    train_X_scaled = scaler.transform(train_X)
    
    np.save(os.path.join(OUTPUT_DIR, f'train_input_config{CONFIG_ID}.npy'), train_X_scaled)
    np.save(os.path.join(OUTPUT_DIR, f'train_target_config{CONFIG_ID}.npy'), train_Y)
    
    print(f"\n>>> Done! Output saved to: {OUTPUT_DIR}")