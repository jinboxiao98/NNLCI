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
RESUME_TRAINING = False 

DATA_DIR = "./data" 
MODEL_FILENAME = 'best_model_lax_resnet.pth' # 新模型名
WEIGHTS_PATH = os.path.join(DATA_DIR, MODEL_FILENAME)

GRID_HF = 800
GRID_L1 = 50
GRID_L2 = 100
WINDOW_SIZE = 5 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Training Lax with ResNet-MLP on: {DEVICE}")

# --- 数据处理 (保持不变: Residual Learning + Z-Score) ---
def load_lax_data(data_dir, mode='train'):
    print(f"📦 Loading {mode} data...")
    search_pattern = os.path.join(data_dir, f"{mode}_P*_{GRID_HF}.dat")
    hf_files = glob.glob(search_pattern)
    
    if not hf_files: raise FileNotFoundError(f"No files found in {data_dir}")

    data_list_hf, data_list_l1, data_list_l2 = [], [], []
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

    raw_target = np.array(data_list_hf)
    raw_in1 = np.array(data_list_l1)
    raw_in2 = np.array(data_list_l2)
    
    # 1. Residual Target
    raw_resid = raw_target - raw_in1 
    
    # 2. Stats
    mean_in = np.mean(raw_in1, axis=(0, 1))
    std_in  = np.std(raw_in1, axis=(0, 1))
    mean_res = np.mean(raw_resid, axis=(0, 1))
    std_res  = np.std(raw_resid, axis=(0, 1))
    
    print(f"📊 Stats:\n   In Mean: {mean_in}\n   Res Mean: {mean_res}")
    
    def z_score(x, m, s): return (x - m) / (s + 1e-6)
    
    in1_norm = z_score(raw_in1, mean_in, std_in)
    in2_norm = z_score(raw_in2, mean_in, std_in)
    target_norm = z_score(raw_resid, mean_res, std_res)
    
    # Windowing
    win_1 = sliding_window_view(in1_norm, window_shape=WINDOW_SIZE, axis=1) 
    win_2 = sliding_window_view(in2_norm, window_shape=WINDOW_SIZE, axis=1) 
    
    trim = WINDOW_SIZE // 2
    target_slice = target_norm[:, trim:-trim, :]
    
    n_features = WINDOW_SIZE * 3 * 2
    
    X_c = win_1.reshape(len(data_list_hf), -1, WINDOW_SIZE * 3)
    X_f = win_2.reshape(len(data_list_hf), -1, WINDOW_SIZE * 3)
    
    X_train = np.concatenate([X_c, X_f], axis=2).reshape(-1, n_features) 
    Y_train = target_slice.reshape(-1, 3)
    
    print(f"✅ Dataset: X={X_train.shape}, Y={Y_train.shape}")
    return torch.FloatTensor(X_train), torch.FloatTensor(Y_train), n_features

# --- 🔥 ResNet 模型定义 ---
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.LayerNorm(dim) # LayerNorm 增加稳定性
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.act(self.fc1(out))
        out = self.bn2(out)
        out = self.fc2(out)
        return out + residual # Skip Connection

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=6, output_dim=3):
        super(ResNet, self).__init__()
        # Input Projection
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Residual Blocks
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        
        # Output Projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

# --- 主程序 ---
if __name__ == '__main__':
    X_train, Y_train, input_dim = load_lax_data(DATA_DIR)
    
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 实例化 ResNet
    # Hidden=256, Blocks=6 (相当于 ~14层深)
    model = ResNet(input_dim=input_dim, hidden_dim=256, num_blocks=6, output_dim=3).to(DEVICE)
    
    if RESUME_TRAINING and os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print("🔄 Resumed weights.")
    else:
        print("✨ Fresh ResNet start.")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # AdamW for better regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150)
    criterion = nn.MSELoss()

    print(f"🚀 Start Training ResNet...")
    model.train()
    
    EPOCHS = 3000
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
    print("✅ ResNet Training Done!")