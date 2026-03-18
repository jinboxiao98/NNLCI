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
MODEL_FILENAME = 'best_model_lax_final.pth' # 读取新模型
WEIGHTS_PATH = os.path.join(DATA_DIR, MODEL_FILENAME)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GRID_HF, GRID_L1, GRID_L2 = 800, 50, 100

# 🔥 配置：Stencil Size = 5
WINDOW_SIZE = 5 

class NeuralNet(nn.Module):
    def __init__(self, input_dim, layers):
        super(NeuralNet, self).__init__()
        all_layers = [input_dim] + layers + [3]
        self.depth = len(all_layers) - 1
        self.actfunc = nn.GELU()
        self.linears = nn.ModuleList()
        for i in range(self.depth):
            layer = nn.Linear(all_layers[i], all_layers[i+1])
            self.linears.append(layer)

    def forward(self, x):
        for i in range(self.depth - 1):
            x = self.actfunc(self.linears[i](x))
        x = self.linears[-1](x)
        return x

def get_scalers(data_dir):
    print("🔄 Calculating Z-Score params...")
    files = glob.glob(os.path.join(data_dir, f"train_P*_{GRID_HF}.dat"))
    if not files: raise FileNotFoundError("No training data found!")

    data_list_l1 = []
    x_target = np.linspace(0, 1, GRID_HF)
    x_l1 = np.linspace(0, 1, GRID_L1)

    for f in files:
        p_str = os.path.basename(f).split('_')[1]
        l1_path = os.path.join(data_dir, f"train_{p_str}_{GRID_L1}.dat")
        
        df_l1 = pd.read_csv(l1_path, sep=r'\s+')
        vars_l1 = df_l1[['den', 'vel', 'pres']].values
        
        interp = np.zeros((GRID_HF, 3))
        for v in range(3):
            interp[:, v] = interp1d(x_l1, vars_l1[:, v], kind='linear', fill_value="extrapolate")(x_target)
        data_list_l1.append(interp)

    all_l1 = np.array(data_list_l1)
    
    mean = np.mean(all_l1, axis=(0, 1))
    std  = np.std(all_l1, axis=(0, 1))
    
    print(f"   Mean: {mean}\n   Std : {std}")
    return mean, std

def evaluate(model, mean, std):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"pred_P*_{GRID_HF}.dat")))
    results = []
    
    x_target = np.linspace(0, 1, GRID_HF)
    x_l1 = np.linspace(0, 1, GRID_L1)
    x_l2 = np.linspace(0, 1, GRID_L2)
    
    model.eval()
    if not os.path.exists("results_lax_win5"): os.makedirs("results_lax_win5")
    
    for hf_path in tqdm(files, desc="Predicting"):
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
            
        # Z-Score
        def z_score(x): return (x - mean) / (std + 1e-6)
        in1_norm = z_score(interp_l1)
        in2_norm = z_score(interp_l2)
        
        # 🔥 Windowing (Size 5)
        win_1 = sliding_window_view(in1_norm, window_shape=WINDOW_SIZE, axis=0).reshape(-1, WINDOW_SIZE*3)
        win_2 = sliding_window_view(in2_norm, window_shape=WINDOW_SIZE, axis=0).reshape(-1, WINDOW_SIZE*3)
        X_input = np.concatenate([win_1, win_2], axis=1) 
        
        # Predict
        with torch.no_grad():
            pred_norm = model(torch.FloatTensor(X_input).to(DEVICE)).cpu().numpy()
            
        pred_inner = pred_norm * (std + 1e-6) + mean
        
        # 🔥 Fill Boundary (Trim = 2)
        trim = WINDOW_SIZE // 2
        pred_full = np.zeros((GRID_HF, 3))
        pred_full[trim:-trim, :] = pred_inner
        
        # 边界直接用插值结果
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
        plt.savefig(f"results_lax_win5/{p_str}.png"); plt.close()

    return pd.DataFrame(results)

if __name__ == "__main__":
    mean, std = get_scalers(DATA_DIR)
    
    # Input Dim = Window(5) * Vars(3) * Fidelity(2) = 30
    model = NeuralNet(WINDOW_SIZE*6, 8*[400]).to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
        print("✅ Model loaded.")
    else: exit()
    
    df = evaluate(model, mean, std)
    df['sort'] = df['Case'].apply(lambda x: float(x.replace('P','')))
    print("\n" + "="*80)
    print(df.sort_values('sort').drop('sort', axis=1).to_string(index=False, float_format=lambda x: "{:.2e}".format(x)))
    print("-" * 80)
    print(f"Global Avg L2: {df['Avg_L2'].mean():.4e}")
    print("="*80)