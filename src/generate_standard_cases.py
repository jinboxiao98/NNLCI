# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 12:18:40 2026

@author: wding64
"""

import os
import numpy as np

# --- 2D Riemann Standard Configurations (Lax & Liu) ---
# 无扰动标准参数
RIEMANN_CONFIGS = {
    3: { 'Q1': [1.5, 0.0, 0.0, 1.5], 'Q2': [0.5323, 1.206, 0.0, 0.3], 'Q3': [0.138, 1.206, 1.206, 0.029], 'Q4': [0.5323, 0.0, 1.206, 0.3] },
    4: { 'Q1': [1.1, 0.0, 0.0, 1.1], 'Q2': [0.5065, 0.8939, 0.0, 0.35], 'Q3': [1.1, 0.8939, 0.8939, 1.1], 'Q4': [0.5065, 0.0, 0.8939, 0.35] },
    6: { 'Q1': [1.0, 0.75, -0.5, 1.0], 'Q2': [2.0, 0.75, 0.5, 1.0], 'Q3': [1.0, -0.75, 0.5, 1.0], 'Q4': [3.0, -0.75, -0.5, 1.0] }
}

# 固定分辨率用于物理验证
RESOLUTION = 800 

# 根据 Config 防止激波跑出边界 (根据你的经验微调)
def get_simulation_time(case_id):
    if case_id == 3: return 0.25 # Modified for validation
    if case_id == 4: return 0.25
    if case_id == 6: return 0.3
    return 0.25

def write_nml(filename, q1, q2, q3, q4, t_end, nx, ny):
    with open(filename, 'w') as f:
        f.write("&CASE_PARAMS\n")
        f.write(f"  q1_r={q1[0]:.6f}, q1_u={q1[1]:.6f}, q1_v={q1[2]:.6f}, q1_p={q1[3]:.6f},\n")
        f.write(f"  q2_r={q2[0]:.6f}, q2_u={q2[1]:.6f}, q2_v={q2[2]:.6f}, q2_p={q2[3]:.6f},\n")
        f.write(f"  q3_r={q3[0]:.6f}, q3_u={q3[1]:.6f}, q3_v={q3[2]:.6f}, q3_p={q3[3]:.6f},\n")
        f.write(f"  q4_r={q4[0]:.6f}, q4_u={q4[1]:.6f}, q4_v={q4[2]:.6f}, q4_p={q4[3]:.6f},\n")
        f.write(f"  DT=0.0001,\n")
        f.write(f"  T_END={t_end},\n")
        f.write(f"  NX={nx}, NY={ny}\n")
        f.write("/\n")

def generate_standard_cases():
    print(f"\n>>> Generating Standard Validation Dataset...")
    root_dir = "DATA_VALIDATION"
    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    for cid, params in RIEMANN_CONFIGS.items():
        # 这里不建立子文件夹，直接放在根目录方便查看
        t_end = get_simulation_time(cid)
        fname = f"input_Config{cid}_std.nml"
        nml_path = os.path.join(root_dir, fname)
        
        # 直接读取参数，不加扰动
        q1 = params['Q1']
        q2 = params['Q2']
        q3 = params['Q3']
        q4 = params['Q4']
        
        write_nml(nml_path, q1, q2, q3, q4, t_end, RESOLUTION, RESOLUTION)
        print(f"    [Config {cid}] Generated standard input -> {nml_path}")

if __name__ == "__main__":
    generate_standard_cases()