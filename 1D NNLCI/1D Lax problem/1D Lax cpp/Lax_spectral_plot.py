import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view

# ==========================================
# 1. 教科书级排版
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "mathtext.fontset": "stix",
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
})

C_HF = "#006400"  # Dark Green
C_LF = "#00008B"  # Navy Blue
C_NN = "#DC143C"  # Crimson

# 尺寸: 3.0 x 1.5
FIG_WIDTH = 3.0   
FIG_HEIGHT = 3 

# 适配 Window=5
WINDOW_SIZE = 5   
MODEL_PATH = "./data/best_model_lax_final.pth" 

GRID_HF = 800
GRID_L1 = 50
GRID_L2 = 100

# ==========================================
# 2. 模型与数据 (保持不变)
# ==========================================
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

def get_plot_data(case_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Searching for Case: {case_id}...")

    # 1. 查找文件
    patterns = [
        f"./data/pred_{case_id}_{GRID_HF}.dat",      
        f"./data/pred_Lax_{case_id}_{GRID_HF}.dat",  
        f"./data/pred_Lax{case_id}_{GRID_HF}.dat",   
        f"./data/train_{case_id}_{GRID_HF}.dat"      
    ]
    f_hf = None
    for p in patterns:
        if os.path.exists(p):
            f_hf = p
            break
    if f_hf is None:
        fallback = glob.glob(f"./data/pred_*{case_id}*_{GRID_HF}.dat")
        if fallback: f_hf = fallback[0]
        else: return None, None, None, None
    print(f"✅ Found: {f_hf}")

    # 2. Scaler
    scaler_files = glob.glob(f"./data/train_*_{GRID_L1}.dat")
    l1_list = []
    x_hf = np.linspace(0, 1, GRID_HF)
    if scaler_files:
        for f in scaler_files:
            try:
                if "Lax" not in f and "P" not in f: continue
                df = pd.read_csv(f, sep=r'\s+')
                interp = np.zeros((GRID_HF, 3))
                cols = ['den', 'vel', 'pres'] if 'den' in df.columns else df.columns[1:4]
                for v in range(3):
                    vals = df[cols[v]] if isinstance(cols[v], str) else df.iloc[:, v+1]
                    interp[:, v] = interp1d(df['x'], vals, kind='linear', fill_value="extrapolate")(x_hf)
                l1_list.append(interp)
            except: continue
    
    if l1_list:
        all_l1 = np.array(l1_list)
        mean = np.mean(all_l1, axis=(0, 1))
        std  = np.std(all_l1, axis=(0, 1))
    else:
        mean = np.array([0.44, 0.28, 0.49]) 
        std  = np.array([0.28, 0.47, 0.38])

    # 3. Model
    input_dim = WINDOW_SIZE * 3 * 2 
    model = NeuralNet(input_dim, 8*[300]).to(device)
    if not os.path.exists(MODEL_PATH): return None, None, None, None
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except:
        try:
            model = NeuralNet(input_dim, 8*[400]).to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        except: return None, None, None, None
    model.eval()

    # 4. LowFi Data
    f_50 = f_hf.replace(str(GRID_HF), str(GRID_L1))
    f_100 = f_hf.replace(str(GRID_HF), str(GRID_L2))
    if not (os.path.exists(f_50) and os.path.exists(f_100)): return None, None, None, None

    df_hf = pd.read_csv(f_hf, sep=r'\s+')
    df_50 = pd.read_csv(f_50, sep=r'\s+')
    df_100 = pd.read_csv(f_100, sep=r'\s+')

    interp_50 = np.zeros((GRID_HF, 3))
    interp_100 = np.zeros((GRID_HF, 3))
    use_iloc = 'den' not in df_hf.columns
    cols = ['den', 'vel', 'pres']

    for v in range(3):
        if use_iloc:
            interp_50[:, v] = interp1d(df_50.iloc[:,0], df_50.iloc[:,v+1], kind='linear', fill_value="extrapolate")(x_hf)
            interp_100[:, v] = interp1d(df_100.iloc[:,0], df_100.iloc[:,v+1], kind='linear', fill_value="extrapolate")(x_hf)
        else:
            interp_50[:, v] = interp1d(df_50['x'], df_50[cols[v]], kind='linear', fill_value="extrapolate")(x_hf)
            interp_100[:, v] = interp1d(df_100['x'], df_100[cols[v]], kind='linear', fill_value="extrapolate")(x_hf)

    def z_score(x): return (x - mean) / (std + 1e-6)
    in_50_norm = z_score(interp_50)
    in_100_norm = z_score(interp_100)

    win_50 = sliding_window_view(in_50_norm, window_shape=WINDOW_SIZE, axis=0).reshape(-1, WINDOW_SIZE*3)
    win_100 = sliding_window_view(in_100_norm, window_shape=WINDOW_SIZE, axis=0).reshape(-1, WINDOW_SIZE*3)
    X_in = np.concatenate([win_50, win_100], axis=1)

    with torch.no_grad():
        pred_norm = model(torch.FloatTensor(X_in).to(device)).cpu().numpy()
    
    pred_val = pred_norm * (std + 1e-6) + mean
    trim = WINDOW_SIZE // 2
    pred_full = np.zeros_like(interp_50)
    pred_full[trim:-trim, :] = pred_val
    pred_full[:trim, :] = interp_100[:trim, :]
    pred_full[-trim:, :] = interp_100[-trim:, :]

    if use_iloc: gt_vals = df_hf.iloc[:, 1:4].values
    else: gt_vals = df_hf[['den','vel','pres']].values

    return x_hf, gt_vals, interp_100, pred_full

# ==========================================
# 3. 核心计算逻辑
# ==========================================
def calculate_spectrum(signal):
    signal_ac = signal - np.mean(signal)
    fft_val = np.fft.rfft(signal_ac)
    power = np.abs(fft_val)**2
    freq = np.arange(len(power))
    return freq, power

# ==========================================
# 4. 绘图主程序 (Final V4)
# ==========================================
def plot_lax_spectral_v4(case_id):
    print(f"🔬 Plotting Final Spectral Analysis (v4) for {case_id}...")
    x, gt, lf, pred = get_plot_data(case_id)
    if x is None: return

    rho_gt = gt[:, 0]
    rho_lf = lf[:, 0]
    rho_nn = pred[:, 0]
    
    freq, spec_gt = calculate_spectrum(rho_gt)
    freq, spec_lf = calculate_spectrum(rho_lf)
    freq, spec_nn = calculate_spectrum(rho_nn)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    l1, = ax.loglog(freq[1:], spec_lf[1:], color=C_LF, linestyle='-.', linewidth=1.0, alpha=0.8, label='Low-Fidelity Input')
    l2, = ax.loglog(freq[1:], spec_gt[1:], color=C_HF, linestyle='-', linewidth=1.0, alpha=0.8, label='High-Fidelity Solution')
    l3, = ax.loglog(freq[1:], spec_nn[1:], color=C_NN, linestyle='--', linewidth=1.0, alpha=0.9, label='NNLCI Prediction')
    
    ax.set_ylabel(r"$P_k$", rotation=0, labelpad=5, ha='right', va='center')
    ax.set_xlabel(r"$k$")
    ax.set_xlim(1, 200) 
    ax.set_ylim(1e-5, 1e5)
    ax.set_title("")
    
    # 🔥 关键调整：图例放左下角 (lower left)
    # 因为对数图通常是左高右低，所以左下角通常是空的
    ax.legend([l2, l3, l1], 
              ['High-Fidelity', 'NNLCI', 'Low-Fidelity'],  
              loc='lower left', 
              frameon=False, 
              fontsize=8.5,
              handlelength=1.5)
    
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.25)
    
    save_path = f"Lax_Spectral_v4_{case_id}.pdf"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    plot_lax_spectral_v4("P2.2")