import os
import subprocess
import numpy as np
import shutil

# --- 配置 ---
# 确保你的 C++ 源代码文件名正确
cpp_filename = "solver.cpp"
executable = "./solver"
output_dir = "data"

# --- Design of Experiments (DoE) ---
# 压力扰动范围: [1.0, 5.0]
doe_plan = {
    # 训练集: 1.0, 1.5, ..., 5.0
    'train': np.arange(1.0, 5.01, 0.5),
    
    # 预测集 (插值): 训练点中间的值
    'pred_interp': [1.25, 2.25, 3.75, 4.25],
    
    # 预测集 (外推): 超出训练范围
    'pred_extrap': [5.5, 6.0]
}

# 网格与精度配置
# 必须与 NNLCI 训练脚本中的 GRID_HF, GRID_L1, GRID_L2 一致
cases = [
    {'grid': 50,  'order': 1}, # Low-Fi 1
    {'grid': 100, 'order': 1}, # Low-Fi 2
    {'grid': 800, 'order': 3}  # High-Fi (GT)
]

# --- 1. 编译 C++ 代码 ---
print(f"🔨 Compiling {cpp_filename}...")
compile_cmd = f"g++ -O3 {cpp_filename} -o {executable}"
ret = os.system(compile_cmd)

if ret != 0:
    print("❌ Error: Compilation failed!")
    exit(1)
else:
    print("✅ Compilation successful.\n")

# --- 2. 创建输出目录 ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 3. 运行 DoE ---
total_runs = sum(len(v) for v in doe_plan.values()) * len(cases)
current_run = 0

print(f"🚀 Starting Data Generation (Total runs: {total_runs})")
print(f"📂 Output will be cleaned to format: {{type}}_P{{val}}_{{grid}}.dat")

for group_name, p_values in doe_plan.items():
    # 确定文件名前缀 ('train' 或 'pred')
    file_prefix = "train" if group_name == 'train' else "pred"
    
    for p_val in p_values:
        for case in cases:
            grid = case['grid']
            order = case['order']
            
            # 格式化 P 值：保留2位小数，确保文件名一致 (e.g., 1.5 -> "1.50")
            p_str = f"{p_val:.2f}"
            
            # 构建 Unique ID (传递给 C++ 的参数)
            # C++ 会输出: data/sol_{unique_id}.dat
            unique_id = f"{file_prefix}_P{p_str}_{grid}"
            
            # 运行 Solver
            # 命令格式: ./solver [nx] [p_left] [order] [output_id]
            cmd = [executable, str(grid), str(p_val), str(order), unique_id]
            
            # print(f"[{current_run+1}/{total_runs}] Running: {unique_id}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Error running case {unique_id}:")
                print(result.stderr)
                continue
            
            # --- 关键修正：重命名文件 ---
            # C++ 默认输出到 data/sol_train_P1.00_50.dat
            # 我们需要重命名为 data/train_P1.00_50.dat 以匹配 NNLCI
            
            src_file = os.path.join(output_dir, f"sol_{unique_id}.dat")
            dst_file = os.path.join(output_dir, f"{unique_id}.dat")
            
            if os.path.exists(src_file):
                os.rename(src_file, dst_file)
                # print(f"   -> Saved: {dst_file}")
            else:
                print(f"⚠️ Warning: Output file not found: {src_file}")
            
            current_run += 1
        
        # 进度条效果
        print(f"✅ Completed P={p_str} group ({current_run}/{total_runs})")

print("\n🎉 All simulations completed.")
print(f"📂 Data is ready in './{output_dir}' for NNLCI training.")