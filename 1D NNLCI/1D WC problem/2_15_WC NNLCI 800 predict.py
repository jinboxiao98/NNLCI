import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view

# --- 配置 ---
ROOT_DIR = "./"
MODEL_FILENAME = 'best_model_800.pth'
WEIGHTS_PATH = os.path.join(ROOT_DIR, 'WC_NNLCI', MODEL_FILENAME)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🚀 Prediction running on: {DEVICE}")

# --- 模型定义 (必须与训练一致) ---
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

# --- 核心工具：重新获取训练时的 Scaling 参数 ---
def recalculate_training_scalers(root_dir):
    """
    重新计算训练集的统计量，确保测试时使用完全相同的变换标准。
    """
    print("🔄 Recalculating scalers from training data to ensure consistency...")
    data_dir = os.path.join(root_dir, 'WC_NNLCI')
    
    # 1. 加载训练集数据
    train_100 = np.load(os.path.join(data_dir, 'training_dataset_100.npy'))
    train_200 = np.load(os.path.join(data_dir, 'training_dataset_200.npy'))
    train_1600 = np.load(os.path.join(data_dir, 'training_dataset_hf.npy')).reshape((401, 1600, 3))
    
    # 训练集索引 (与 train.py 一致)
    indices = np.arange(0, 400, 10)
    train_100_src = train_100[indices]
    train_200_src = train_200[indices]
    train_800_target = train_1600[indices, ::2, :] 

    # 插值
    x_100 = np.linspace(0, 1, 100)
    x_200 = np.linspace(0, 1, 200)
    x_800 = np.linspace(0, 1, 800)
    
    input_100_interp = np.zeros_like(train_800_target)
    input_200_interp = np.zeros_like(train_800_target)
    
    for i in range(len(indices)):
        for v in range(3):
            f1 = interp1d(x_100, train_100_src[i, :, v], kind='linear', fill_value="extrapolate")
            input_100_interp[i, :, v] = f1(x_800)
            f2 = interp1d(x_200, train_200_src[i, :, v], kind='linear', fill_value="extrapolate")
            input_200_interp[i, :, v] = f2(x_800)

    # 1. 确定 Shift
    min_val = min(train_800_target.min(), input_100_interp.min(), input_200_interp.min())
    if min_val <= -2.0:
        shift_val = abs(min_val) + 1.0
    else:
        shift_val = 2.0
    
    print(f"   -> Found Shift: {shift_val}")

    # 2. Log 变换
    def transform(x): return np.log10(x + shift_val + 1e-6)
    input_100_interp = transform(input_100_interp)
    
    # 3. 确定 lb, ub
    ub = np.max(input_100_interp, axis=(0, 1))
    lb = np.min(input_100_interp, axis=(0, 1))
    
    return shift_val, lb, ub

# --- 预测单个 Case 的函数 ---
def predict_case(model, case_index, root_dir, shift_val, lb, ub):
    data_dir = os.path.join(root_dir, 'WC_NNLCI')
    
    # 加载单个测试样本
    raw_100 = np.load(os.path.join(data_dir, 'training_dataset_100.npy'))[case_index] 
    raw_200 = np.load(os.path.join(data_dir, 'training_dataset_200.npy'))[case_index] 
    raw_1600 = np.load(os.path.join(data_dir, 'training_dataset_hf.npy')).reshape((401, 1600, 3))[case_index]
    
    # Ground Truth (800 grid)
    truth_800 = raw_1600[::2, :] 

    # 1. 插值
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

    # 2. 变换
    def transform(x): return np.log10(x + shift_val + 1e-6)
    
    interp_100_log = transform(interp_100)
    interp_200_log = transform(interp_200)
    
    # 3. Scaling
    diff = ub - lb
    diff[diff == 0] = 1.0
    def scale(x): return 2 * (x - lb) / diff - 1
    
    in100_norm = scale(interp_100_log)
    in200_norm = scale(interp_200_log)
    
    # 4. 构造 Window
    win_100 = sliding_window_view(in100_norm, window_shape=3, axis=0) 
    win_200 = sliding_window_view(in200_norm, window_shape=3, axis=0)
    
    feat_100 = win_100.reshape(798, 9)
    feat_200 = win_200.reshape(798, 9)
    
    X_input = np.concatenate([feat_100, feat_200], axis=1)
    
    # 5. 预测
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_input).to(DEVICE)
        pred_norm = model(X_tensor).cpu().numpy() 
        
    # 6. 反变换
    pred_log = (pred_norm + 1) * diff / 2 + lb
    pred_final = 10**pred_log - (shift_val + 1e-6)
    
    # 7. 填充边界 (使用插值结果填充首尾)
    full_pred = np.zeros((800, 3))
    full_pred[1:-1, :] = pred_final
    full_pred[0, :] = interp_100[0, :]
    full_pred[-1, :] = interp_100[-1, :]
    
    return full_pred, truth_800, interp_100

# --- 🔥 新增：误差计算函数 ---
def compute_metrics(pred, target):
    """
    计算 Relative L1 和 Relative L2 Error
    Pred, Target shape: [N, 3] (Density, Velocity, Pressure)
    """
    # 防止除以 0
    eps = 1e-8
    
    # axis=0 表示沿着 800 个网格点求和/求范数，保留 3 个变量维度
    # 1. Relative L1 Norm: sum(|pred - true|) / sum(|true|)
    l1_diff = np.sum(np.abs(pred - target), axis=0)
    l1_target = np.sum(np.abs(target), axis=0)
    rel_l1 = l1_diff / (l1_target + eps)

    # 2. Relative L2 Norm: sqrt(sum((pred - true)^2)) / sqrt(sum(true^2))
    l2_diff = np.sqrt(np.sum((pred - target)**2, axis=0))
    l2_target = np.sqrt(np.sum(target**2, axis=0))
    rel_l2 = l2_diff / (l2_target + eps)
    
    return rel_l1, rel_l2

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 准备参数
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Error: Model weights not found at {WEIGHTS_PATH}")
        exit()

    shift_val, lb, ub = recalculate_training_scalers(ROOT_DIR)

    # 2. 加载模型
    layers = [18] + 10*[300] + [3]
    model = NeuralNet(layers).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    print("✅ Model loaded successfully.")

    # 3. 选择 Case
    TEST_CASE_IDX = 55 
    print(f"🔮 Predicting Case {TEST_CASE_IDX}...")
    
    # 获取预测结果
    pred_800, truth_800, input_100 = predict_case(model, TEST_CASE_IDX, ROOT_DIR, shift_val, lb, ub)

    # --- 🔥 计算并打印误差 ---
    rel_l1, rel_l2 = compute_metrics(pred_800, truth_800)
    
    var_names = ['Density (ρ)', 'Velocity (u)', 'Pressure (p)']
    
    print("\n" + "="*50)
    print(f"📊 Error Metrics for Case {TEST_CASE_IDX}")
    print("="*50)
    print(f"{'Variable':<15} | {'Rel L1 Error':<15} | {'Rel L2 Error':<15}")
    print("-" * 50)
    
    for i in range(3):
        print(f"{var_names[i]:<15} | {rel_l1[i]:.6f}        | {rel_l2[i]:.6f}")
        
    print("-" * 50)
    print(f"{'Average':<15} | {np.mean(rel_l1):.6f}        | {np.mean(rel_l2):.6f}")
    print("="*50 + "\n")

    # 5. 绘图
    x_axis = np.linspace(0, 1, 800)
    plt.figure(figsize=(18, 5))
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(x_axis, truth_800[:, i], 'k-', label='Ground Truth', linewidth=2.5, alpha=0.3)
        plt.plot(x_axis, input_100[:, i], 'g--', label='Low-Fi Input', linewidth=1.5, alpha=0.6)
        plt.plot(x_axis, pred_800[:, i], 'r-', label='NNLCI Prediction', linewidth=1.5)
        
        # 将 L2 Error 标注在标题上
        plt.title(f"{var_names[i]}\nRel L2 Error: {rel_l2[i]:.2e}")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    save_name = f"prediction_case_{TEST_CASE_IDX}.png"
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"🖼️ Plot saved to {save_name}")
    plt.show()