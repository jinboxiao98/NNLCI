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
# 1. 教科书级绘图配置 (Textbook Configuration)
# ==========================================
# 这一步确保生成的图无需后期处理即可直接用于 LaTeX/Word
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

# --- 学术配色 (深色系，高对比度) ---
# High-Fidelity: Forest Green (权威、基准)
C_GT = "#006400" 
# Low-Fidelity: Navy Blue (冷色、背景)
C_LF = "#00008B"
# NNLCI: Crimson Red (暖色、强调)
C_NN = "#DC143C"

# --- 布局尺寸 ---
# 宽度 4.75 inch (标准单栏/紧凑双栏宽度)
# 高度 2.0 inch (增高，使激波结构更舒展)
FIG_WIDTH = 4.75 
FIG_HEIGHT = 2.0  
MODEL_PATH = "./data/best_model_sod_final.pth"

# --- 物理参数 (必须与训练一致) ---
GRID_HF = 800
GRID_L1 = 50   # Scaler基准
GRID_L2 = 100
WINDOW_SIZE = 3

# ==========================================
# 2. 模型定义
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

# ==========================================
# 3. 数据处理 (核心逻辑)
# ==========================================
def get_plot_data(case_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- A. 正确计算 Scaler (过滤 Lax 数据) ---
    train_files = glob.glob(f"./data/train_P*_{GRID_L1}.dat")
    l1_list = []
    x_hf = np.linspace(0, 1, GRID_HF)
    
    for f in train_files:
        try:
            # 过滤掉 P > 5.0 的 Lax 数据，防止污染 Scaler
            p_val = float(os.path.basename(f).split('_')[1].replace('P', ''))
            if p_val > 5.0: continue
            
            df = pd.read_csv(f, sep=r'\s+')
            # 兼容列名
            cols = ['den', 'vel', 'pres'] if 'den' in df.columns else df.columns[1:4]
            interp = np.zeros((GRID_HF, 3))
            for v in range(3):
                vals = df[cols[v]] if isinstance(cols[v], str) else df.iloc[:, v+1]
                interp[:, v] = interp1d(df['x'], vals, kind='linear', fill_value="extrapolate")(x_hf)
            l1_list.append(interp)
        except: continue
            
    all_l1 = np.array(l1_list)
    mean = np.mean(all_l1, axis=(0, 1))
    std  = np.std(all_l1, axis=(0, 1))

    # --- B. 加载模型 ---
    input_dim = WINDOW_SIZE * 3 * 2 
    model = NeuralNet(input_dim, 8*[400]).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    else:
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return None, None, None, None
    model.eval()

    # --- C. 加载 Case 数据 ---
    base_path = f"./data/pred_{case_id}"
    if not os.path.exists(f"{base_path}_{GRID_HF}.dat"): 
        base_path = f"./data/train_{case_id}"
    
    f_hf = f"{base_path}_{GRID_HF}.dat"
    f_50 = f"{base_path}_{GRID_L1}.dat"
    f_100 = f"{base_path}_{GRID_L2}.dat"

    if not os.path.exists(f_hf): return None, None, None, None

    df_hf = pd.read_csv(f_hf, sep=r'\s+')
    df_50 = pd.read_csv(f_50, sep=r'\s+')
    df_100 = pd.read_csv(f_100, sep=r'\s+')

    # --- D. 预处理与推理 ---
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
    # 边界填充
    pred_full[:trim, :] = interp_100[:trim, :]
    pred_full[-trim:, :] = interp_100[-trim:, :]

    return x_hf, df_hf[['den','vel','pres']].values, interp_100, pred_full

# ==========================================
# 4. 绘图主程序
# ==========================================
def plot_main_prediction(case_id):
    print(f"🎨 Generating Main Plot for {case_id}...")
    x, gt, lf, pred = get_plot_data(case_id)
    if x is None: 
        print(f"❌ Data not found for {case_id}")
        return

    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))
    titles = ["Density", "Velocity", "Pressure"]
    
    for i in range(3):
        ax = axes[i]
        
        # --- 线条绘制 (注意顺序和 zorder) ---
        
        # 1. Low-Fidelity (底层): 蓝色 点划线
        # 作为背景参考，不需要太抢眼
        l1, = ax.plot(x, lf[:, i], color=C_LF, linestyle='-.', linewidth=1.0, 
                      label='Low-Fidelity Input', zorder=1)
        
        # 2. High-Fidelity (中层): 绿色 实线
        # 这是真值，用实线表示权威性。不透明。
        l2, = ax.plot(x, gt[:, i], color=C_GT, linestyle='-', linewidth=1.0, 
                      label='High-Fidelity Solution', zorder=2)
        
        # 3. NNLCI (顶层): 红色 虚线
        # 骑在绿线上。虚线的空隙会透出底下的绿色，形成漂亮的红绿相间效果。
        l3, = ax.plot(x, pred[:, i], color=C_NN, linestyle='--', linewidth=1.0, 
                      label='NNLCI Prediction', zorder=3)
        
        # --- 刻度与标签 ---
        ax.set_title(titles[i], pad=6, fontsize=8) # 标题距离
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        
        # Y轴刻度: 自动选择最简洁的 4-5 个刻度
        ax.locator_params(axis='y', nbins=5)
        
        # 只有最左边的图显示 Y 轴数值? 
        # 教科书通常建议每个图都保留刻度，方便读数。保持现状即可。

    # --- 布局调整 ---
    # bottom=0.28 给下方的 Legend 留出足够空间
    # wspace=0.25 让三个子图之间不至于太挤
    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.28, wspace=0.25)
    
    # --- Legend (图例) ---
    handles = [l2, l3, l1] # 顺序: 真值, 预测, 输入
    labels = [h.get_label() for h in handles]
    
    fig.legend(handles, labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), # 坐标 (x, y) 相对整个图
               ncol=3,                     # 横排三个
               frameon=False,              # 无边框，融入背景
               columnspacing=1.5,          # 列间距
               handlelength=2.0)           # 线条画长一点，方便看清虚实

    # 保存
    filename = f"Sod_Main_{case_id}.pdf"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.show()
    print(f"✅ Saved: {filename}")
    plt.close()

if __name__ == "__main__":
    # 建议使用 P2.25 作为主展示 Case
    plot_main_prediction("P1.25")