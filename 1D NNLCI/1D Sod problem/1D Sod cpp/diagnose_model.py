# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 23:10:53 2026

@author: wding64
"""

import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pandas as pd
from scipy.interpolate import interp1d

# --- 配置 ---
DATA_DIR = "./data"
MODEL_PATH = os.path.join(DATA_DIR, 'best_model_sod.pth')
GRID_HF = 800
GRID_L1 = 50
GRID_L2 = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 模型定义 ---
class NeuralNet(nn.Module):
    def __init__(self, layers):
        super(NeuralNet, self).__init__()
        self.depth = len(layers) - 1
        self.actfunc = nn.Tanh()
        self.linears = nn.ModuleList()
        for i in range(self.depth):
            layer = nn.Linear(layers[i], layers[i+1])
            self.linears.append(layer)

    def forward(self, x):
        for i in range(self.depth - 1):
            x = self.actfunc(self.linears[i](x))
        x = self.linears[-1](x)
        return x

def diagnose():
    print("🩺 Starting NNLCI Diagnosis...\n")

    # 1. 检查模型文件和权重
    print(f"1️⃣  Checking Model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found!")
        return
    
    try:
        model = NeuralNet([18] + 10*[300] + [3]).to(DEVICE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        
        # 检查第一层权重
        first_layer_weight = state_dict['linears.0.weight'].cpu().numpy()
        print(f"   -> First Layer Stats: Min={first_layer_weight.min():.4f}, Max={first_layer_weight.max():.4f}, Mean={first_layer_weight.mean():.4f}")
        
        if np.isnan(first_layer_weight).any():
            print("❌ CRITICAL: Model weights contain NaN! The model was saved during a crashed training session.")
            print("👉 Solution: Re-run training (train_sod.py) from scratch.")
            return
        
        if np.allclose(first_layer_weight, 0, atol=1e-6):
            print("⚠️ Warning: Model weights are very close to zero. Training might have failed to converge.")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    print("✅ Model weights look numerically valid (not NaN).\n")

    # 2. 检查 Scaler 参数一致性
    print("2️⃣  Checking Scalers (Data Consistency)")
    files = glob.glob(os.path.join(DATA_DIR, f"train_P*_{GRID_HF}.dat"))
    if not files:
        print("❌ Error: No training data found to calculate scalers.")
        return
    
    print(f"   -> Found {len(files)} training files.")
    
    # 快速计算 Shift, LB, UB
    data_list_l1 = []
    global_min = float('inf')
    x_target = np.linspace(0, 1, GRID_HF)
    x_l1 = np.linspace(0, 1, GRID_L1)
    
    for f in files:
        # P value
        p_str = os.path.basename(f).split('_')[1]
        l1_path = os.path.join(DATA_DIR, f"train_{p_str}_{GRID_L1}.dat")
        
        df_hf = pd.read_csv(f, sep='\s+')
        df_l1 = pd.read_csv(l1_path, sep='\s+')
        
        global_min = min(global_min, df_hf.values.min(), df_l1.values.min())
        
        # Interp
        vars_l1 = df_l1[['den', 'vel', 'pres']].values
        interp = np.zeros((GRID_HF, 3))
        for v in range(3):
            interp[:, v] = interp1d(x_l1, vars_l1[:, v], kind='linear', fill_value="extrapolate")(x_target)
        data_list_l1.append(interp)
        
    all_l1 = np.array(data_list_l1)
    
    if global_min <= -2.0: shift = abs(global_min) + 1.0
    else: shift = 2.0
    
    def transform(x): return np.log10(x + shift + 1e-6)
    l1_trans = transform(all_l1)
    ub = np.max(l1_trans, axis=(0, 1))
    lb = np.min(l1_trans, axis=(0, 1))
    
    print(f"   -> Calculated Shift: {shift:.4f}")
    print(f"   -> Calculated LB: {lb}")
    print(f"   -> Calculated UB: {ub}")
    
    if np.isnan(lb).any() or np.isnan(ub).any():
        print("❌ CRITICAL: Scalers are NaN! Your training data still contains NaNs.")
        return
    print("✅ Scalers calculated successfully.\n")

    # 3. 模拟预测 (Dry Run)
    print("3️⃣  Dry Run Prediction (Test Input Range)")
    model.eval()
    
    # 造一个伪数据 (使用第一组训练数据的插值结果)
    sample_input_l1 = all_l1[0] # [800, 3]
    
    # 归一化
    diff = ub - lb
    diff[diff == 0] = 1.0
    
    def scale(x): return 2 * (x - lb) / diff - 1
    
    sample_norm = scale(transform(sample_input_l1))
    print(f"   -> Input to Model (Normalized): Min={sample_norm.min():.4f}, Max={sample_norm.max():.4f}")
    
    if sample_norm.min() < -1.5 or sample_norm.max() > 1.5:
        print("⚠️ Warning: Input values deviate significantly from [-1, 1]. Normalization might be skewed.")
    
    # 构造 Window 输入
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(sample_norm, window_shape=3, axis=0).reshape(-1, 9)
    # 模拟双路输入 (L1+L2)，这里简单复制一份用于测试
    X_test = np.concatenate([win, win], axis=1) # [798, 18]
    X_tensor = torch.FloatTensor(X_test).to(DEVICE)
    
    with torch.no_grad():
        output = model(X_tensor).cpu().numpy()
        
    print(f"   -> Model Output (Normalized): Min={output.min():.4f}, Max={output.max():.4f}, Mean={output.mean():.4f}")
    
    if np.allclose(output, output[0,0], atol=1e-3):
        print("❌ CRITICAL: Model output is constant/flat! The model is not learning.")
    elif np.isnan(output).any():
        print("❌ CRITICAL: Model output contains NaNs!")
    else:
        print("✅ Model output looks dynamic.")

if __name__ == "__main__":
    diagnose()