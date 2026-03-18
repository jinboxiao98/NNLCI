import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view

# ==========================================
# 1. 教科书级绘图配置 (From plot_sod.py)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,               # 正文 8pt
    "axes.labelsize": 9,          # 轴标签 9pt
    "axes.titlesize": 10,         # 标题 10pt
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,       # 线宽 1.5
    "xtick.direction": "in",      # 刻度朝里
    "ytick.direction": "in",
    "figure.dpi": 300,            # 300 DPI
    "savefig.bbox": "tight",      
    "axes.grid": False,           # 无网格
})

# 配色 (Sod Style)
C_GT = "#006400"  # High-Fidelity: Forest Green
C_LF = "#00008B"  # Low-Fidelity: Navy Blue
C_NN = "#DC143C"  # NNLCI: Crimson Red

# 尺寸 (Sod Style)
FIG_WIDTH = 4.75 
FIG_HEIGHT = 2.0

# 路径配置
ROOT_DIR = "./"
MODEL_FILENAME = 'best_model_800.pth'
WEIGHTS_PATH = os.path.join(ROOT_DIR, 'WC_NNLCI', MODEL_FILENAME)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 模型定义 (WC specific: Tanh)
# ==========================================
class NeuralNet(nn.Module):
    def __init__(self, layers):
        super(NeuralNet, self).__init__()
        self.depth = len(layers) - 1
        self.actfunc = nn.Tanh() # WC uses Tanh
        self.linears = nn.ModuleList()
        for i in range(self.depth):
            layer = nn.Linear(layers[i], layers[i+1])
            self.linears.append(layer)

    def forward(self, x):
        for i in range(self.depth - 1):
            x = self.actfunc(self.linears[i](x))
        x = self.linears[-1](x)
        return x

# ==========================================
# 3. 数据处理 (From WC script)
# ==========================================
def recalculate_training_scalers(root_dir):
    print("🔄 Calculating scalers...")
    data_dir = os.path.join(root_dir, 'WC_NNLCI')
    
    try:
        train_100 = np.load(os.path.join(data_dir, 'training_dataset_100.npy'))
        train_1600 = np.load(os.path.join(data_dir, 'training_dataset_hf.npy')).reshape((401, 1600, 3))
    except FileNotFoundError:
        print("❌ Error: .npy files not found in ./WC_NNLCI/")
        return None, None, None

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
    if min_val <= -2.0:
        shift_val = abs(min_val) + 1.0
    else:
        shift_val = 2.0
    
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
    
    interp_100_log = transform(interp_100)
    interp_200_log = transform(interp_200)
    
    diff = ub - lb
    diff[diff == 0] = 1.0
    def scale(x): return 2 * (x - lb) / diff - 1
    
    in100_norm = scale(interp_100_log)
    in200_norm = scale(interp_200_log)
    
    win_100 = sliding_window_view(in100_norm, window_shape=3, axis=0) 
    win_200 = sliding_window_view(in200_norm, window_shape=3, axis=0)
    
    feat_100 = win_100.reshape(798, 9)
    feat_200 = win_200.reshape(798, 9)
    
    X_input = np.concatenate([feat_100, feat_200], axis=1)
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_input).to(DEVICE)
        pred_norm = model(X_tensor).cpu().numpy() 
        
    pred_log = (pred_norm + 1) * diff / 2 + lb
    pred_final = 10**pred_log - (shift_val + 1e-6)
    
    full_pred = np.zeros((800, 3))
    full_pred[1:-1, :] = pred_final
    full_pred[0, :] = interp_100[0, :]
    full_pred[-1, :] = interp_100[-1, :]
    
    return full_pred, truth_800, interp_100

# ==========================================
# 4. 绘图主程序
# ==========================================
if __name__ == "__main__":
    # 1. Scalers
    shift_val, lb, ub = recalculate_training_scalers(ROOT_DIR)
    if shift_val is None:
        print("Skipping plotting due to missing data.")
        exit()

    # 2. Model
    layers = [18] + 10*[300] + [3]
    model = NeuralNet(layers).to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
        print("✅ Model loaded.")
    else:
        print(f"❌ Model not found: {WEIGHTS_PATH}")
        exit()

    # 3. Predict Case 55
    CASE_IDX = 55
    print(f"Plotting Case {CASE_IDX} with Sod Style...")
    pred, gt, lf = predict_case(model, CASE_IDX, ROOT_DIR, shift_val, lb, ub)
    
    x_axis = np.linspace(0, 1, 800)

    # 4. Plotting (Sod Style)
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))
    titles = ["Density", "Velocity", "Pressure"]
    
    for i in range(3):
        ax = axes[i]
        
        # Low-Fi: Blue Dash-Dot
        l1, = ax.plot(x_axis, lf[:, i], color=C_LF, linestyle='-.', linewidth=1.0, 
                      label='Low-Fidelity Input', zorder=1)
        
        # High-Fi: Green Solid
        l2, = ax.plot(x_axis, gt[:, i], color=C_GT, linestyle='-', linewidth=1.0, 
                      label='High-Fidelity Solution', zorder=2)
        
        # NNLCI: Red Dashed
        l3, = ax.plot(x_axis, pred[:, i], color=C_NN, linestyle='--', linewidth=1.0, 
                      label='NNLCI Prediction', zorder=3)
        
        ax.set_title(titles[i], pad=6, fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.locator_params(axis='y', nbins=5)
        
    # Layout Adjustments (Exact match to plot_sod.py)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.28, wspace=0.25)
    
    # Legend
    handles = [l2, l3, l1]
    labels = ["High-Fidelity Solution", "NNLCI Prediction", "Low-Fidelity Input"]
    
    fig.legend(handles, labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), 
               ncol=3, 
               frameon=False,
               columnspacing=1.5,
               handlelength=2.0)

    save_name = f"WC_Main_SodStyle_Case{CASE_IDX}.pdf"
    plt.savefig(save_name, dpi=600, bbox_inches='tight', pad_inches=0.02) 
    print(f"✅ Saved plot to {save_name}")