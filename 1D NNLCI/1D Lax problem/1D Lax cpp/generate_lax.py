import os
import subprocess
import numpy as np
import shutil

# --- 配置 ---
cpp_filename = "solver_lax.cpp"
executable = "./solver_lax"
output_dir = "data"

# --- Lax Problem DoE ---
# Range: P_Left in [2.0, 6.0]
doe_plan = {
    'train': np.arange(2.0, 6.01, 0.4), # 2.0, 2.5, ... 6.0
    'pred_interp': [2.2, 3.8, 5.4],
    'pred_extrap': [6.4, 6.8]
}

cases = [
    {'grid': 50,  'order': 1},
    {'grid': 100, 'order': 1},
    {'grid': 800, 'order': 3}
]

# --- 1. Compile ---
print(f"🔨 Compiling {cpp_filename}...")
ret = os.system(f"g++ -O3 {cpp_filename} -o {executable}")
if ret != 0:
    print("❌ Compilation Failed!"); exit(1)

# --- 2. Run ---
if not os.path.exists(output_dir): os.makedirs(output_dir)

total_runs = sum(len(v) for v in doe_plan.values()) * len(cases)
curr = 0

print(f"🚀 Starting Lax Data Generation (Total runs: {total_runs})")
print(f"📂 Output format: train/pred_P{{val}}_{{grid}}.dat")

for group, p_vals in doe_plan.items():
    # 🔥 修改点：移除 'lax_' 前缀，直接使用 'train' 或 'pred'
    # 这样文件名格式为：train_P2.00_50.dat
    if group == 'train':
        prefix = "train"
    else:
        # group 是 pred_interp 或 pred_extrap
        prefix = "pred"
    
    for p in p_vals:
        for case in cases:
            grid = case['grid']
            order = case['order']
            p_str = f"{p:.2f}"
            
            # Unique ID 传递给 C++ (e.g., train_P3.50_50)
            unique_id = f"{prefix}_P{p_str}_{grid}"
            
            # C++ outputs: data/sol_{unique_id}.dat
            cmd = [executable, str(grid), str(p), str(order), unique_id]
            subprocess.run(cmd, capture_output=True)
            
            # Rename: remove 'sol_' prefix
            src = os.path.join(output_dir, f"sol_{unique_id}.dat")
            dst = os.path.join(output_dir, f"{unique_id}.dat")
            
            if os.path.exists(src):
                os.rename(src, dst)
            else:
                print(f"⚠️ Missing output file: {src}")
                
            curr += 1
        print(f"✅ Completed Lax P={p_str} ({curr}/{total_runs})")

print("🎉 Lax Data Generation Complete.")