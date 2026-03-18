import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ==========================================
# 🔥 用户手动配置区 (MANUAL CONFIGURATION)
# ==========================================

# 1. 放大视野范围 [x_min, x_max, y_min, y_max]
# 您可以根据生成的图片，看着坐标轴刻度来微调这里
ZOOM_RANGES = {
    'Density':  [0.60, 0.80,  3,  8],  # 关注双峰
    'Velocity': [0.67, 0.87, 6,  15],  # 关注单峰
    'Pressure': [0.60, 0.70,  250, 410.0]   # 关注压力峰
}

# 2. 放大框在图中的显示尺寸 (百分比)
INSET_WIDTH  = "40%"  # 宽度
INSET_HEIGHT = "40%"  # 高度

# 3. 放大框的位置 (1=右上角, 2=左上角, 3=左下角, 4=右下角)
# 建议：Density放左上(2)避开波峰，其他放右上(1)
INSET_LOCATIONS = {
    'Density':  2,
    'Velocity': 2,
    'Pressure': 2
}

# ==========================================
# 1. 教科书级绘图配置
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,               
    "axes.labelsize": 9,          
    "axes.titlesize": 10,         
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
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
FIG_HEIGHT = 2.0

ROOT_DIR = "./"
MODEL_FILENAME = 'best_model_800.pth'
WEIGHTS_PATH = os.path.join(ROOT_DIR, 'WC_NNLCI', MODEL_FILENAME)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 模型与数据 (保持不变)
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
# 3. 绘图主程序 (手动控制版)
# ==========================================
if __name__ == "__main__":
    shift_val, lb, ub = recalculate_training_scalers(ROOT_DIR)
    if shift_val is None: exit()
    layers = [18] + 10*[300] + [3]
    model = NeuralNet(layers).to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    else: exit()

    CASE_IDX = 55
    print(f"🎨 Plotting Manual Zoom for Case {CASE_IDX}...")
    pred, gt, lf = predict_case(model, CASE_IDX, ROOT_DIR, shift_val, lb, ub)
    x_axis = np.linspace(0, 1, 800)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))
    titles = ["Density", "Velocity", "Pressure"]
    
    for i in range(3):
        ax = axes[i]
        
        # Plot Main
        l1, = ax.plot(x_axis, lf[:, i], color=C_LF, linestyle='-.', linewidth=1.0, label='Low-Fidelity')
        l2, = ax.plot(x_axis, gt[:, i], color=C_GT, linestyle='-', linewidth=1.0, label='High-Fidelity')
        l3, = ax.plot(x_axis, pred[:, i], color=C_NN, linestyle='--', linewidth=1.0, label='NNLCI')
        
        ax.set_title(titles[i], pad=6, fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.locator_params(axis='y', nbins=5)

        # 🔥 获取手动配置
        key = titles[i]
        x1, x2, y1, y2 = ZOOM_RANGES[key]
        loc_pos = INSET_LOCATIONS[key]
        
        # Inset Plot
        axins = inset_axes(ax, width=INSET_WIDTH, height=INSET_HEIGHT, loc=loc_pos)
        
        axins.plot(x_axis, lf[:, i], color=C_LF, linestyle='-.', linewidth=1.0)
        axins.plot(x_axis, gt[:, i], color=C_GT, linestyle='-', linewidth=1.0)
        axins.plot(x_axis, pred[:, i], color=C_NN, linestyle='--', linewidth=1.0)
        
        # 应用手动范围
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        
        axins.set_xticks([]) 
        axins.set_yticks([])
        
        for spine in axins.spines.values():
            spine.set_edgecolor('0.4')
        
        # 连线逻辑：根据位置自动选择角落
        if loc_pos == 2: # 左上角 -> 连右下
            loc1, loc2 = 3, 4
        elif loc_pos == 1: # 右上角 -> 连左下
            loc1, loc2 = 2, 4
        else:
            loc1, loc2 = 2, 4 # 默认

        mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.4", lw=0.8)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.28, wspace=0.25)
    
    fig.legend([l2, l3, l1], ["High-Fidelity Solution", "NNLCI Prediction", "Low-Fidelity Input"], 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), 
               ncol=3, 
               frameon=False,
               columnspacing=1.5)

    save_name = f"WC_Zoom_Manual_Case{CASE_IDX}.pdf"
    plt.savefig(save_name, dpi=600, bbox_inches='tight', pad_inches=0.02)
    print(f"✅ Saved Manual Zoom Plot: {save_name}")