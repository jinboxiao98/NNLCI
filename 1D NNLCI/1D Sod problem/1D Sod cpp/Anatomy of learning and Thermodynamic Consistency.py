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
# 1. 极致排版配置 (Textbook Configuration)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    # 🔥 关键修改：字体全部设为 8.5
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    # 🔥 关键修改：数学字体设为 stix (Times-like)，确保公式也是 Times New Roman 风格
    "mathtext.fontset": "stix",
    
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
})

# 配色
C_HF = "#006400"  # Dark Green
C_LF = "#00008B"  # Navy Blue
C_NN = "#DC143C"  # Crimson

# 尺寸 (单栏)
FIG_WIDTH = 3.5   
FIG_HEIGHT = 2.6

MODEL_PATH = "./data/best_model_sod_final.pth"
GRID_HF = 800
GRID_L1 = 50
GRID_L2 = 100
WINDOW_SIZE = 3

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
    
    train_files = glob.glob(f"./data/train_P*_{GRID_L1}.dat")
    l1_list = []
    x_hf = np.linspace(0, 1, GRID_HF)
    
    for f in train_files:
        try:
            p_val = float(os.path.basename(f).split('_')[1].replace('P', ''))
            if p_val > 5.0: continue
            df = pd.read_csv(f, sep=r'\s+')
            interp = np.zeros((GRID_HF, 3))
            cols = ['den', 'vel', 'pres'] if 'den' in df.columns else df.columns[1:4]
            for v in range(3):
                vals = df[cols[v]] if isinstance(cols[v], str) else df.iloc[:, v+1]
                interp[:, v] = interp1d(df['x'], vals, kind='linear', fill_value="extrapolate")(x_hf)
            l1_list.append(interp)
        except: continue
            
    if not l1_list: return None, None, None, None

    all_l1 = np.array(l1_list)
    mean = np.mean(all_l1, axis=(0, 1))
    std  = np.std(all_l1, axis=(0, 1))

    input_dim = WINDOW_SIZE * 3 * 2 
    model = NeuralNet(input_dim, 8*[300]).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except:
        model = NeuralNet(input_dim, 8*[400]).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    base_path = f"./data/pred_{case_id}"
    if not os.path.exists(f"{base_path}_{GRID_HF}.dat"): base_path = f"./data/train_{case_id}"
    f_hf = f"{base_path}_{GRID_HF}.dat"
    f_50 = f"{base_path}_{GRID_L1}.dat"
    f_100 = f"{base_path}_{GRID_L2}.dat"

    if not os.path.exists(f_hf): return None, None, None, None

    df_hf = pd.read_csv(f_hf, sep=r'\s+')
    df_50 = pd.read_csv(f_50, sep=r'\s+')
    df_100 = pd.read_csv(f_100, sep=r'\s+')

    interp_50 = np.zeros((GRID_HF, 3))
    interp_100 = np.zeros((GRID_HF, 3))
    cols = ['den', 'vel', 'pres']
    for v in range(3):
        col = cols[v]
        interp_50[:, v] = interp1d(df_50['x'], df_50[col], kind='linear', fill_value="extrapolate")(x_hf)
        interp_100[:, v] = interp1d(df_100['x'], df_100[col], kind='linear', fill_value="extrapolate")(x_hf)

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

    return x_hf, df_hf[['den','vel','pres']].values, interp_100, pred_full

# ==========================================
# 3. Plot 1: Anatomy of Learning (Final)
# ==========================================
def plot_residual(case_id):
    print(f"📊 Generating Final Residual Plot for {case_id}...")
    x, gt, lf, pred = get_plot_data(case_id)
    if x is None: return

    # Density
    idx = 0 
    
    target_correction = gt[:, idx] - lf[:, idx]
    predicted_correction = pred[:, idx] - lf[:, idx]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # 1. True Error (Background)
    ax.fill_between(x, target_correction, color='gray', alpha=1.0, linewidth=1.0,
                    label='High-Fidelity $-$ Low-Fidelity')
    
    # 2. Predicted Correction (Foreground)
    ax.plot(x, predicted_correction, color=C_NN, linestyle='--', linewidth=1.0,
            label='NNLCI Correction')
    
    # y=0 Line
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    
    # --- Refined Labels (Times New Roman 8.5pt) ---
    ax.set_xlabel(r"$x$")
    
    # 🔥 关键调整：横置 Y Label，减小 labelpad (15 -> 5)，让标签紧贴轴，节省空间
    ax.set_ylabel(r"$\Delta \rho$", rotation=0, labelpad=8, ha='right', va='center')
    
    ax.set_xlim(0.4, 0.95)
    
    y_min = min(target_correction.min(), predicted_correction.min())
    y_max = max(target_correction.max(), predicted_correction.max())
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin)
    
    # Legend
    ax.legend(frameon=False, loc='upper left')
    
    plt.tight_layout()
    save_path = f"Sod_Analysis_Residual_{case_id}_Final.pdf"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    print(f"✅ Saved: {save_path}")

# ==========================================
# 4. Plot 2: Phase Space (Final)
# ==========================================
def plot_phase_space(case_id):
    print(f"🌀 Generating Final Phase Space Plot for {case_id}...")
    x, gt, lf, pred = get_plot_data(case_id)
    if x is None: return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # 1. High-Fidelity
    ax.plot(gt[:, 0], gt[:, 2], color=C_HF, linestyle='-', linewidth=2.0, alpha=1.0, 
            label='High-Fidelity Simulation')
    
    # 2. NNLCI
    ax.plot(pred[:, 0], pred[:, 2], color=C_NN, linestyle='--', linewidth=1.5, 
            label='NNLCI Prediction')
    
    # --- Refined Labels ---
    ax.set_xlabel(r"$\rho$") 
    # 横置 Y Label，同样减小 labelpad
    ax.set_ylabel(r"$P$", rotation=0, labelpad=8, ha='right', va='center') 
    
    # Legend
    ax.legend(frameon=False, loc='best')
    
    plt.tight_layout()
    save_path = f"Sod_Analysis_PhaseSpace_{case_id}_Final.pdf"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    case = "P1.25"
    plot_residual(case)
    plot_phase_space(case)