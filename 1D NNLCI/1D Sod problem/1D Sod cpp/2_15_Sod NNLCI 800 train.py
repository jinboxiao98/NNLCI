import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view

# --- 配置 ---
NUM_WORKERS = 0 
BATCH_SIZE = 2048
USE_AMP = False 
PIN_MEMORY = True 
RESUME_TRAINING = False

DATA_DIR = "./data" 
MODEL_FILENAME = 'best_model_sod_final.pth' # 新模型名
WEIGHTS_PATH = os.path.join(DATA_DIR, MODEL_FILENAME)

GRID_HF = 800
GRID_L1 = 50
GRID_L2 = 100
WINDOW_SIZE = 3 # 保持为 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Training (Final Opt: NoLog, Z-Score, GELU) on: {DEVICE}")

# --- 数据处理 ---
def load_sod_data(data_dir, mode='train'):
    print(f"📦 Loading {mode} data...")
    search_pattern = os.path.join(data_dir, f"{mode}_P*_{GRID_HF}.dat")
    hf_files = glob.glob(search_pattern)
    
    if not hf_files:
        raise FileNotFoundError("No training files found.")

    data_list_hf = []
    data_list_l1 = []
    data_list_l2 = []

    x_target = np.linspace(0, 1, GRID_HF)
    x_l1 = np.linspace(0, 1, GRID_L1)
    x_l2 = np.linspace(0, 1, GRID_L2)

    for hf_path in hf_files:
        filename = os.path.basename(hf_path)
        parts = filename.replace('.dat', '').split('_')
        p_str = parts[1]
        
        l1_path = os.path.join(data_dir, f"{mode}_{p_str}_{GRID_L1}.dat")
        l2_path = os.path.join(data_dir, f"{mode}_{p_str}_{GRID_L2}.dat")
        
        if not os.path.exists(l1_path) or not os.path.exists(l2_path): continue
            
        df_hf = pd.read_csv(hf_path, sep=r'\s+')
        df_l1 = pd.read_csv(l1_path, sep=r'\s+')
        df_l2 = pd.read_csv(l2_path, sep=r'\s+')
        
        vars_hf = df_hf[['den', 'vel', 'pres']].values
        vars_l1 = df_l1[['den', 'vel', 'pres']].values
        vars_l2 = df_l2[['den', 'vel', 'pres']].values
        
        interp_l1 = np.zeros_like(vars_hf)
        interp_l2 = np.zeros_like(vars_hf)
        
        for v in range(3):
            f1 = interp1d(x_l1, vars_l1[:, v], kind='linear', fill_value="extrapolate")
            interp_l1[:, v] = f1(x_target)
            f2 = interp1d(x_l2, vars_l2[:, v], kind='linear', fill_value="extrapolate")
            interp_l2[:, v] = f2(x_target)
            
        data_list_hf.append(vars_hf)
        data_list_l1.append(interp_l1)
        data_list_l2.append(interp_l2)

    # shape: [N_cases, 800, 3]
    train_target = np.array(data_list_hf)
    in1 = np.array(data_list_l1)
    in2 = np.array(data_list_l2)
    
    # 🔥 优化 1 & 2: 移除 Log, 使用 Z-Score Standardization
    # 计算全局 Mean 和 Std (基于 Low-Fi 数据)
    # axis=(0, 1) 表示对 (Case, GridPoint) 维度求均值，保留 (Variable) 维度
    mean = np.mean(in1, axis=(0, 1)) # [3]
    std  = np.std(in1, axis=(0, 1))  # [3]
    
    print(f"📊 Z-Score Stats:\n   Mean: {mean}\n   Std : {std}")
    
    def z_score(x, m, s): return (x - m) / (s + 1e-6)
    
    target_norm = z_score(train_target, mean, std)
    in1_norm = z_score(in1, mean, std)
    in2_norm = z_score(in2, mean, std)
    
    # Windowing (Size 3)
    win_1 = sliding_window_view(in1_norm, window_shape=WINDOW_SIZE, axis=1) 
    win_2 = sliding_window_view(in2_norm, window_shape=WINDOW_SIZE, axis=1) 
    
    # Target 切片匹配
    trim = WINDOW_SIZE // 2
    target_slice = target_norm[:, trim:-trim, :]
    
    # Flatten
    # Input features: Window(3) * Var(3) * Fidelity(2) = 18 features
    n_features = WINDOW_SIZE * 3 * 2
    
    X_c = win_1.reshape(len(data_list_hf), -1, WINDOW_SIZE * 3)
    X_f = win_2.reshape(len(data_list_hf), -1, WINDOW_SIZE * 3)
    
    X_train = np.concatenate([X_c, X_f], axis=2).reshape(-1, n_features) 
    Y_train = target_slice.reshape(-1, 3)
    
    print(f"✅ Dataset: X={X_train.shape}, Y={Y_train.shape}")
    return torch.FloatTensor(X_train), torch.FloatTensor(Y_train), n_features

# --- 模型定义 ---
class NeuralNet(nn.Module):
    def __init__(self, input_dim, layers):
        super(NeuralNet, self).__init__()
        all_layers = [input_dim] + layers + [3]
        self.depth = len(all_layers) - 1
        
        # 🔥 优化 3: 使用 GELU 激活函数
        self.actfunc = nn.GELU() 
        
        self.linears = nn.ModuleList()
        for i in range(self.depth):
            layer = nn.Linear(all_layers[i], all_layers[i+1])
            # Kaiming Init 适合 GELU/ReLU
            nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')
            nn.init.zeros_(layer.bias.data)
            self.linears.append(layer)

    def forward(self, x):
        for i in range(self.depth - 1):
            x = self.actfunc(self.linears[i](x))
        x = self.linears[-1](x)
        return x

# --- 主程序 ---
if __name__ == '__main__':
    X_train, Y_train, input_dim = load_sod_data(DATA_DIR)
    
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)

    hidden_layers = 8 * [400] # 保持深度
    model = NeuralNet(input_dim, hidden_layers).to(DEVICE)
    
    if RESUME_TRAINING and os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print("🔄 Resumed weights.")
    else:
        print("✨ Fresh start.")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    criterion = nn.MSELoss()

    print(f"🚀 Start Training (Win={WINDOW_SIZE}, GELU, Z-Score)...")
    model.train()
    
    EPOCHS = 4000
    for epoch in tqdm(range(EPOCHS)):
        epoch_loss = 0.0
        batches = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
        
        avg_loss = epoch_loss / batches
        scheduler.step(avg_loss)

        if epoch % 500 == 0:
            tqdm.write(f"Epoch {epoch} | Loss: {avg_loss:.4e} | LR: {optimizer.param_groups[0]['lr']:.1e}")
            torch.save(model.state_dict(), WEIGHTS_PATH)
            
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print("✅ Done!")