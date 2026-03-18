import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle

# ==========================================
# 1. 极致排版配置 (与 plot_sod.py 保持完全一致)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,               # 正文 8pt
    "axes.labelsize": 9,          # 轴标签略大
    "axes.titlesize": 10,         # 标题 10pt
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,       # 标准线宽
    "xtick.direction": "in",      # 刻度朝里
    "ytick.direction": "in",
    "figure.dpi": 300,            # 印刷级分辨率
    "savefig.bbox": "tight",      # 自动裁剪白边
    "axes.grid": False,           # 无网格 (最简风格)
})

# 配色
C_HF = "#006400"  # Dark Green (Solid)
C_LF = "#00008B"  # Navy Blue (Dash-Dot)
C_NN = "#DC143C"  # Crimson (Dashed)

# 布局
FIG_WIDTH = 4.75  
FIG_HEIGHT = 2.0
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
    # 注意：这里使用了 8*[400] 以匹配您上传的文件，请根据实际训练模型调整
    model = NeuralNet(input_dim, 8*[400]).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    else: return None, None, None, None
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
# 3. 绘图主程序 (Manual Connectors)
# ==========================================
def plot_final_one(case_id):
    print(f"🎨 Generating Final Textbook Figure with Custom Connectors for {case_id}...")
    x, gt, lf, pred = get_plot_data(case_id)
    if x is None: return

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))
    titles = ["Density", "Velocity", "Pressure"]
    
    # Zoom-in 坐标 (微调版)
    zoom_coords = [
        # Density
        [0.65, 0.75, 0.25, 0.45],   
        # Velocity (稍微包含一点负值)
        [0.82, 0.96, -0.07, 0.25],    
        # Pressure (包含 Jump)
        [0.82, 0.92, 0.05, 0.45]     
    ]

    for i in range(3):
        ax = axes[i]
        
        # Main Plot
        l1, = ax.plot(x, lf[:, i], color=C_LF, linestyle='-.', linewidth=1.0, label='Low-Fidelity Input')
        l2, = ax.plot(x, gt[:, i], color=C_HF, linestyle='-', linewidth=1.0, label='High-Fidelity Solution')
        l3, = ax.plot(x, pred[:, i], color=C_NN, linestyle='--', linewidth=1.0, label='NNLCI Prediction')
        
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlim(0, 1.0)
        ax.locator_params(axis='y', nbins=5)
        # 统一标题样式：不加粗，pad=6，fontsize=8
        ax.set_title(titles[i], pad=6, fontsize=8)
        
        # --- 🔍 Custom Zoom-in Logic ---
        x1, x2, y1, y2 = zoom_coords[i]
        
        # 1. 创建 Inset Axes (右上角)
        axins = inset_axes(ax, width="45%", height="45%", loc=1)
        
        axins.plot(x, lf[:, i], color=C_LF, linestyle='-.', linewidth=1.0)
        axins.plot(x, gt[:, i], color=C_HF, linestyle='-', linewidth=1.2)
        axins.plot(x, pred[:, i], color=C_NN, linestyle='--', linewidth=1.2)
        
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        
        # 2. 手动画连接线 (Projection Style)
        # 先在主图上画出 ROI 框
        rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec="0.5", lw=0.8, ls='-')
        ax.add_patch(rect)
        
        # 连接线 1: 左上 -> 左下
        con1 = ConnectionPatch(xyA=(x1, y2), xyB=(x1, y1), coordsA="data", coordsB="data",
                               axesA=ax, axesB=axins, color="0.5", lw=0.8, ls="-", alpha=0.6)
        fig.add_artist(con1)
        
        # 连接线 2: 右上 -> 右下
        con2 = ConnectionPatch(xyA=(x2, y2), xyB=(x2, y1), coordsA="data", coordsB="data",
                               axesA=ax, axesB=axins, color="0.5", lw=0.8, ls="-", alpha=0.6)
        fig.add_artist(con2)

        # 给 Inset 加个边框颜色，使其突显
        for spine in axins.spines.values():
            spine.set_edgecolor('0.3')

    # 布局调整：与 plot_sod.py 完全一致
    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.28, wspace=0.25)
    
    fig.legend([l2, l3, l1], ['High-Fidelity Solution', 'NNLCI Prediction', 'Low-Fidelity Input'], 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.0), 
               ncol=3, 
               frameon=False,
               columnspacing=1.0)

    save_path = f"Sod_Final_{case_id}_ManualConnect.pdf"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    plot_final_one("P1.25")