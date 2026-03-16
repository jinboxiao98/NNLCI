import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time

# ==========================================
# 1. 全局配置
# ==========================================
# 基础路径配置
ROOT_DIR = "./"
DATA_DIR = os.path.join(ROOT_DIR, "NNLCI_Data/Config3_Stencils/")
MODEL_DIR = os.path.join(ROOT_DIR, "NNLCI_Models/")
OUTPUT_DIR = os.path.join(ROOT_DIR, "NNLCI_Output/")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练超参数
CONFIG_ID = "3"
BATCH_SIZE = 65536      # 批次大小 (显存足够大可保持，不够则减半)
EPOCHS = 500            # 训练轮数
LEARNING_RATE = 1e-4    # 初始学习率
WEIGHT_DECAY = 1e-9     # L2 正则化

# [修改] 数据加载加速配置
NUM_WORKERS = 12        # 增加 Worker 数量
PERSISTENT_WORKERS = True # 保持 Worker 进程存活，减少创建开销

# 自动选择设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {DEVICE}")

# ==========================================
# 2. 数据集类 (内存映射版)
# ==========================================
class NNLCIDataset(Dataset):
    def __init__(self, input_path, target_path):
        """
        使用 mmap_mode='r' 读取大文件，避免一次性加载到 RAM。
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        self.input_data = np.load(input_path, mmap_mode='r')
        self.target_data = np.load(target_path, mmap_mode='r')
        
        assert self.input_data.shape[0] == self.target_data.shape[0], "Size mismatch!"
        self.length = self.input_data.shape[0]
        
        print(f"Dataset loaded: {self.length} samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.from_numpy(self.input_data[idx].copy()).float()
        y = torch.from_numpy(self.target_data[idx].copy()).float()
        return x, y

# ==========================================
# 3. 神经网络模型 (Deep MLP)
# ==========================================
class NNLCI_Net(nn.Module):
    def __init__(self, input_dim=72, output_dim=4, hidden_layers=None):
        super(NNLCI_Net, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [600] * 10  # 默认 10 层 600 节点
            
        layers = []
        # Input Layer
        in_dim = input_dim
        
        # Hidden Layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh()) # Tanh 激活
            in_dim = h_dim
            
        # Output Layer (Regression)
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Xavier Initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. 训练与验证逻辑
# ==========================================
def train():
    # --- 文件路径 ---
    train_input_file = os.path.join(DATA_DIR, f"train_input_config{CONFIG_ID}.npy")
    train_target_file = os.path.join(DATA_DIR, f"train_target_config{CONFIG_ID}.npy")
    
    # --- 加载数据 ---
    print(">>> Initializing Dataset...")
    dataset = NNLCIDataset(train_input_file, train_target_file)
    
    # 划分训练集和验证集 (90% / 10%)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # [修改] DataLoader 性能优化参数
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS # 关键加速点
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    # --- 初始化模型 ---
    # [修改] 10 层 x 600 节点
    deep_layers = [600] * 10
    print(f">>> Initializing Deep Network: {deep_layers}")
    
    model = NNLCI_Net(input_dim=72, output_dim=4, hidden_layers=deep_layers).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # [注意] 已移除 verbose=True 以兼容新版 PyTorch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    criterion = nn.MSELoss()
    
    # 混合精度训练 (加速)
    scaler = torch.cuda.amp.GradScaler()

    print(f">>> Start Training Config {CONFIG_ID} for {EPOCHS} Epochs...")
    
    model_save_path = os.path.join(MODEL_DIR, f"nnlci_config{CONFIG_ID}_deep_latest.pth")
    best_loss = float('inf')

    start_total = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        start_epoch = time.time()
        
        # 预取数据通常由 DataLoader 自动处理，无需手动 prefetcher
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = criterion(pred, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # 裁剪前必须先 unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - start_epoch
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.2e} | Val Loss: {avg_val_loss:.2e} | Time: {epoch_time:.1f}s | LR: {current_lr:.1e}")
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)

    total_time = time.time() - start_total
    print(f"\n>>> Training Complete. Total Time: {total_time/60:.1f} mins")
    print(f">>> Best Model Saved to: {model_save_path}")

if __name__ == "__main__":
    train()