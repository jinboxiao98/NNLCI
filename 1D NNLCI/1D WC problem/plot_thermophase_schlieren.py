import torch
import torch.nn as nn
import numpy as np
import os
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

C_GT = "#006400"
C_LF = "#00008B"
C_NN = "#DC143C"

FIG_WIDTH = 4.75 
FIG_HEIGHT = 2.6

ROOT_DIR = "./"
MODEL_FILENAME = 'best_model_800.pth'
WEIGHTS_PATH = os.path.join(ROOT_DIR, 'WC_NNLCI', MODEL_FILENAME)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 模型与数据 (复用之前逻辑)
# ==========================================
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

def recalculate_training_scalers(root_dir):
    data_dir = os.path.join(root_dir, 'WC_NNLCI')
    try:
        train_100 = np.load(os.path.join(data_dir, 'training_dataset_100.npy'))
        train_1600 = np.load(os.path.join(data_dir, 'training_dataset_hf.npy')).reshape((401, 1600, 3))
    except: return None, None, None

    indices = np.arange(0, 400, 10)
    train_100_src = train_100[indices]
    train_800_target = train_1600[indices, ::2, :] 
    x_100 = np.linspace(0, 1, 100)
    x_800 = np.linspace(0, 1, 800)
    input_100_interp = np.zeros_like(train_800_target)
    for i in range(len(indices)):
        for v in range(3):
            f1 = interp1d(x_100, train_100_src[i, :, v], kind='linear', fill_value="extrapolate")
            input_100_interp[i, :, v] = f1(x_800)
    min_val = min(train_800_target.min(), input_100_interp.min())
    shift_val = abs(min_val) + 1.0 if min_val <= -2.0 else 2.0
    def transform(x): return np.log10(x + shift_val + 1e-6)
    input_100_interp = transform(input_100_interp)
    ub = np.max(input_100_interp, axis=(0, 1))
    lb = np.min(input_100_interp, axis=(0, 1))
    return shift_val, lb, ub

def predict_case(model, case_index, root_dir, shift_val, lb, ub):
    data_dir = os.path.join(root_dir, 'WC_NNLCI')
    raw_100 = np.load(os.path.join(data_dir, 'training_dataset_100.npy'))[case_index] 
    raw_200 = np.load(os.path.join(data_dir, 'training_dataset_200.npy'))[case_index] 
    raw_1600 = np.load(os.path.join(data_dir, 'training_dataset_hf.npy')).reshape((401, 1600, 3))[case_index]
    truth_800 = raw_1600[::2, :] 
    x_100 = np.linspace(0, 1, 100)
    x_200 = np.linspace(0, 1, 200)
    x_800 = np.linspace(0, 1, 800)
    interp_100 = np.zeros((800, 3))
    interp_200 = np.zeros((800, 3))
    for v in range(3):
        f1 = interp1d(x_100, raw_100[:, v], kind='linear', fill_value="extrapolate")
        interp_100[:, v] = f1(x_800)
        f2 = interp1d(x_200, raw_200[:, v], kind='linear', fill_value="extrapolate")
        interp_200[:, v] = f2(x_800)
    def transform(x): return np.log10(x + shift_val + 1e-6)
    in100_norm = 2 * (transform(interp_100) - lb) / (ub - lb + 1e-8) - 1
    in200_norm = 2 * (transform(interp_200) - lb) / (ub - lb + 1e-8) - 1
    win_100 = sliding_window_view(in100_norm, window_shape=3, axis=0).reshape(798, 9)
    win_200 = sliding_window_view(in200_norm, window_shape=3, axis=0).reshape(798, 9)
    X_input = np.concatenate([win_100, win_200], axis=1)
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.FloatTensor(X_input).to(DEVICE)).cpu().numpy()
    pred_log = (pred_norm + 1) * (ub - lb + 1e-8) / 2 + lb
    pred_final = 10**pred_log - (shift_val + 1e-6)
    full_pred = np.zeros((800, 3))
    full_pred[1:-1, :] = pred_final
    full_pred[0, :] = interp_100[0, :]
    full_pred[-1, :] = interp_100[-1, :]
    return full_pred, truth_800, interp_100

# ==========================================
# 3. 计算导数 (Numerical Schlieren)
# ==========================================
def calculate_gradient(y, x):
    # 使用中心差分计算梯度
    return np.gradient(y, x, edge_order=2)

# ==========================================
# 4. 绘图主程序 (Swapped & Rotated Labels)
# ==========================================
def plot_wc_analysis_swapped(case_id=55):
    print(f"🔬 Plotting WC Analysis (Swapped) for Case {case_id}...")
    
    # 1. 准备数据
    shift_val, lb, ub = recalculate_training_scalers(ROOT_DIR)
    if shift_val is None: return
    layers = [18] + 10*[300] + [3]
    model = NeuralNet(layers).to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    else: return

    pred, gt, lf = predict_case(model, case_id, ROOT_DIR, shift_val, lb, ub)
    x_axis = np.linspace(0, 1, 800)

    # 提取变量
    rho_gt, rho_lf, rho_nn = gt[:, 0], lf[:, 0], pred[:, 0]
    pres_gt, pres_lf, pres_nn = gt[:, 2], lf[:, 2], pred[:, 2]

    # 计算梯度 (Schlieren)
    grad_gt = np.abs(calculate_gradient(rho_gt, x_axis))
    grad_lf = np.abs(calculate_gradient(rho_lf, x_axis))
    grad_nn = np.abs(calculate_gradient(rho_nn, x_axis))

    # 2. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # --- Plot A (Left): Phase Space Trajectory (P vs Rho) ---
    ax1 = axes[0] # 现在是左图
    
    ax1.plot(rho_lf, pres_lf, color=C_LF, linestyle='-.', linewidth=1.0, alpha=0.9, label='Low-Fidelity')
    ax1.plot(rho_gt, pres_gt, color=C_GT, linestyle='-', linewidth=1.0, alpha=0.9, label='High-Fidelity')
    ax1.plot(rho_nn, pres_nn, color=C_NN, linestyle='--', linewidth=1.0, alpha=0.9, label='NNLCI')
    
    ax1.set_xlabel(r"$\rho$")
    # 🔥 Y Label: 符号 P, 横向, 靠右对齐
    ax1.set_ylabel(r"$P$", rotation=0, labelpad=5, ha='right', va='center')
    ax1.set_title("Thermodynamic Phase Space", pad=6, fontsize=8.5)

    # --- Plot B (Right): Numerical Schlieren (Density Gradient) ---
    ax2 = axes[1] # 现在是右图
    
    # Low-Fi: 矮胖的峰
    ax2.plot(x_axis, grad_lf, color=C_LF, linestyle='-.', linewidth=1.0, alpha=0.8, label='Low-Fidelity')
    # High-Fi: 高瘦的峰
    ax2.plot(x_axis, grad_gt, color=C_GT, linestyle='-', linewidth=1.0, alpha=0.8, label='High-Fidelity')
    # NNLCI: 尝试重合
    ax2.plot(x_axis, grad_nn, color=C_NN, linestyle='--', linewidth=1.0, alpha=0.9, label='NNLCI')
    
    ax2.set_xlabel(r"$x$")
    # 🔥 Y Label: 符号 |\nabla \rho|, 横向, 靠右对齐
    ax2.set_ylabel(r"$|\nabla \rho|$", rotation=0, labelpad=5, ha='right', va='center')
    ax2.set_xlim(0, 1.0)
    
    # 限制 Y 轴
    y_limit = np.percentile(grad_gt, 99.5) * 1.2
    ax2.set_ylim(0, y_limit)
    
    ax2.set_title("Numerical Schlieren (Shock Resolution)", pad=6, fontsize=8.5)
    
    # 统一图例
    lines, labels = ax1.get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.25, wspace=0.3)
    
    save_path = f"WC_Analysis_Swapped_Case{case_id}.pdf"
    plt.savefig(save_path)
    print(f"✅ Saved Analysis Plot: {save_path}")

if __name__ == "__main__":
    plot_wc_analysis_swapped(55)