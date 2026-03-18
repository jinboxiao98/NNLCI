# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 23:59:48 2026

@author: wding64
"""

import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

DATA_DIR = "./data"
MODEL_FILENAME = 'best_model_lax_resnet.pth'
WEIGHTS_PATH = os.path.join(DATA_DIR, MODEL_FILENAME)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GRID_HF, GRID_L1, GRID_L2 = 800, 50, 100
WINDOW_SIZE = 5 

# --- 🔥 ResNet 模型定义 (必须一致) ---
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.LayerNorm(dim)
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.act(self.fc1(out))
        out = self.bn2(out)
        out = self.fc2(out)
        return out + residual

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=6, output_dim=3):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks: x = block(x)
        x = self.output_layer(x)
        return x

# --- Scaler Utils ---
def get_scalers(data_dir):
    print("🔄 Calculating Residual Scalers...")
    files = glob.glob(os.path.join(data_dir, f"train_P*_{GRID_HF}.dat"))
    if not files: raise FileNotFoundError("No training data found!")

    data_list_hf = []
    data_list_l1 = []
    x_target = np.linspace(0, 1, GRID_HF)
    x_l1 = np.linspace(0, 1, GRID_L1)

    for f in files:
        p_str = os.path.basename(f).split('_')[1]
        l1_path = os.path.join(data_dir, f"train_{p_str}_{GRID_L1}.dat")
        
        df_hf = pd.read_csv(f, sep=r'\s+')
        df_l1 = pd.read_csv(l1_path, sep=r'\s+')
        
        vars_hf = df_hf[['den', 'vel', 'pres']].values
        vars_l1 = df_l1[['den', 'vel', 'pres']].values
        
        interp = np.zeros((GRID_HF, 3))
        for v in range(3):
            interp[:, v] = interp1d(x_l1, vars_l1[:, v], kind='linear', fill_value="extrapolate")(x_target)
        
        data_list_hf.append(vars_hf)
        data_list_l1.append(interp)

    raw_target = np.array(data_list_hf)
    raw_in1 = np.array(data_list_l1)
    
    raw_resid = raw_target - raw_in1
    
    mean_in = np.mean(raw_in1, axis=(0, 1))
    std_in  = np.std(raw_in1, axis=(0, 1))
    mean_res = np.mean(raw_resid, axis=(0, 1))
    std_res  = np.std(raw_resid, axis=(0, 1))
    
    return mean_in, std_in, mean_res, std_res

def evaluate(model, mean_in, std_in, mean_res, std_res):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"pred_P*_{GRID_HF}.dat")))
    results = []
    
    x_target = np.linspace(0, 1, GRID_HF)
    x_l1 = np.linspace(0, 1, GRID_L1)
    x_l2 = np.linspace(0, 1, GRID_L2)
    
    model.eval()
    if not os.path.exists("results_lax_resnet"): os.makedirs("results_lax_resnet")
    
    for hf_path in tqdm(files, desc="Predicting ResNet"):
        p_str = os.path.basename(hf_path).split('_')[1]
        l1_path = os.path.join(DATA_DIR, f"pred_{p_str}_{GRID_L1}.dat")
        l2_path = os.path.join(DATA_DIR, f"pred_{p_str}_{GRID_L2}.dat")
        
        try:
            truth = pd.read_csv(hf_path, sep=r'\s+')[['den', 'vel', 'pres']].values
            raw_l1 = pd.read_csv(l1_path, sep=r'\s+')[['den', 'vel', 'pres']].values
            raw_l2 = pd.read_csv(l2_path, sep=r'\s+')[['den', 'vel', 'pres']].values
        except: continue
            
        interp_l1 = np.zeros((GRID_HF, 3))
        interp_l2 = np.zeros((GRID_HF, 3))
        for v in range(3):
            f1 = interp1d(x_l1, raw_l1[:, v], kind='linear', fill_value="extrapolate")
            interp_l1[:, v] = f1(x_target)
            f2 = interp1d(x_l2, raw_l2[:, v], kind='linear', fill_value="extrapolate")
            interp_l2[:, v] = f2(x_target)
            
        # Input Z-Score
        def z_score(x, m, s): return (x - m) / (s + 1e-6)
        in1_norm = z_score(interp_l1, mean_in, std_in)
        in2_norm = z_score(interp_l2, mean_in, std_in)
        
        win_1 = sliding_window_view(in1_norm, window_shape=WINDOW_SIZE, axis=0).reshape(-1, WINDOW_SIZE*3)
        win_2 = sliding_window_view(in2_norm, window_shape=WINDOW_SIZE, axis=0).reshape(-1, WINDOW_SIZE*3)
        X_input = np.concatenate([win_1, win_2], axis=1) 
        
        # Predict Residuals
        with torch.no_grad():
            pred_res_norm = model(torch.FloatTensor(X_input).to(DEVICE)).cpu().numpy()
            
        # Denormalize Residuals
        pred_res = pred_res_norm * (std_res + 1e-6) + mean_res
        
        # Recon: Pred = LowFi + Residual
        trim = WINDOW_SIZE // 2
        base_low_fi = interp_l1[trim:-trim, :]
        pred_inner = base_low_fi + pred_res
        
        pred_full = np.zeros((GRID_HF, 3))
        pred_full[trim:-trim, :] = pred_inner
        pred_full[:trim, :] = interp_l1[:trim, :]
        pred_full[-trim:, :] = interp_l1[-trim:, :]
        
        eps = 1e-8
        l2 = np.sqrt(np.sum((pred_full - truth)**2, axis=0)) / (np.sqrt(np.sum(truth**2, axis=0)) + eps)
        l1 = np.sum(np.abs(pred_full - truth), axis=0) / (np.sum(np.abs(truth), axis=0) + eps)
        
        results.append({
            'Case': p_str,
            'Avg_L2': np.mean(l2), 'Rho_L2': l2[0], 'Vel_L2': l2[1], 'Pres_L2': l2[2],
            'Avg_L1': np.mean(l1)
        })
        
        plt.figure(figsize=(15, 4))
        for v in range(3):
            plt.subplot(1,3,v+1)
            plt.plot(x_target, truth[:,v], 'k-', alpha=0.3)
            plt.plot(x_target, interp_l1[:,v], 'g--', alpha=0.5)
            plt.plot(x_target, pred_full[:,v], 'r-')
            plt.title(f"{['Den','Vel','Pres'][v]} L2:{l2[v]:.2e}")
        plt.tight_layout()
        plt.savefig(f"results_lax_resnet/{p_str}.png"); plt.close()

    return pd.DataFrame(results)

if __name__ == "__main__":
    mean_in, std_in, mean_res, std_res = get_scalers(DATA_DIR)
    
    input_dim = WINDOW_SIZE * 6 # 30
    model = ResNet(input_dim=input_dim).to(DEVICE)
    
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
        print("✅ ResNet Model loaded.")
    
    df = evaluate(model, mean_in, std_in, mean_res, std_res)
    df['sort'] = df['Case'].apply(lambda x: float(x.replace('P','')))
    print("\n" + "="*80)
    print(df.sort_values('sort').drop('sort', axis=1).to_string(index=False, float_format=lambda x: "{:.2e}".format(x)))
    print("-" * 80)
    print(f"Global Avg L2: {df['Avg_L2'].mean():.4e}")
    print("="*80)