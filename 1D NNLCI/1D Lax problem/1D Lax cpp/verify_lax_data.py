# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 23:41:38 2026

@author: wding64
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 配置 ---
DATA_DIR = "./data"
IMG_DIR = "./verify_lax_imgs"

# 阈值设置
CRITICAL_NEG = -0.1 # 允许轻微的数值振荡，但不能太离谱
CHECK_COUNT = 3     # 随机抽取几个 Case 画图检查

def check_and_plot():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: {DATA_DIR} not found.")
        return

    # 1. 扫描所有 .dat 文件
    # 兼容两种命名: train_P*.dat (新) 或 lax_train_P*.dat (旧)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_P*.dat")))
    
    if not files:
        print("❌ No data files found.")
        return

    print(f"🔍 Found {len(files)} files. Starting verification...")
    
    bad_files = []
    valid_groups = {} # Key: P_Value, Value: {grid: filepath}

    # --- 第一步：数值健康检查 ---
    for f in files:
        fname = os.path.basename(f)
        try:
            # 解析 P 值和 Grid
            # 格式预期: [prefix]_P{val}_{grid}.dat
            parts = fname.replace('.dat', '').split('_')
            grid = int(parts[-1])
            p_val = parts[-2] # P3.50
            
            # 读取数据
            df = pd.read_csv(f, sep=r'\s+')
            
            # 1. NaN / Inf 检查
            if df.isnull().values.any() or np.isinf(df.values).any():
                print(f"❌ [CORRUPT] {fname} contains NaN or Inf!")
                bad_files.append(fname)
                continue
                
            # 2. 物理合理性检查 (Lax 问题中 Density 和 Pressure 必须 > 0)
            min_den = df['den'].min()
            min_pres = df['pres'].min()
            
            if min_den < CRITICAL_NEG or min_pres < CRITICAL_NEG:
                print(f"⚠️ [PHYSICS FAIL] {fname}: Negative Den/Pres detected! (Min Den={min_den:.4f})")
                bad_files.append(fname)
                continue
                
            # 收集有效文件用于画图
            if p_val not in valid_groups: valid_groups[p_val] = {}
            valid_groups[p_val][grid] = f

        except Exception as e:
            print(f"❌ [READ ERROR] {fname}: {e}")

    print("\n" + "="*40)
    print(f"📊 Verification Summary")
    print("="*40)
    
    if bad_files:
        print(f"🚫 Found {len(bad_files)} BAD files. Check log above.")
    else:
        print(f"✅ All {len(files)} files passed numerical checks (No NaN, No Severe Negatives).")

    # --- 第二步：视觉检查 (Plotting) ---
    if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)
    
    # 随机抽取几个 P 值进行画图
    sample_ps = random.sample(list(valid_groups.keys()), min(CHECK_COUNT, len(valid_groups)))
    # 确保包含标准 Lax Case (P3.50 附近) 如果存在
    for p in valid_groups.keys():
        if "3.5" in p: 
            if p not in sample_ps: sample_ps[0] = p
            break
            
    print(f"\n🎨 Plotting {len(sample_ps)} cases to '{IMG_DIR}' for visual inspection...")

    for p_str in sample_ps:
        grids_dict = valid_groups[p_str]
        
        # 我们希望对比 High-Fi (800) 和 Low-Fi (50)
        if 800 not in grids_dict: continue
        
        # 读取数据
        df_hf = pd.read_csv(grids_dict[800], sep=r'\s+')
        
        plt.figure(figsize=(15, 4))
        
        # 绘制变量: Density, Velocity, Pressure
        vars_map = [('den', 'Density'), ('vel', 'Velocity'), ('pres', 'Pressure')]
        
        for idx, (col, label) in enumerate(vars_map):
            plt.subplot(1, 3, idx+1)
            
            # 画 High-Fi
            plt.plot(df_hf['x'], df_hf[col], 'k-', linewidth=2, label='High-Fi (800)')
            
            # 画 Low-Fi (如果存在)
            if 50 in grids_dict:
                df_lf = pd.read_csv(grids_dict[50], sep=r'\s+')
                plt.plot(df_lf['x'], df_lf[col], 'r--', linewidth=1.5, alpha=0.7, label='Low-Fi (50)')
            
            plt.title(f"{label} (P_L={p_str.replace('P','')})")
            plt.grid(True, linestyle=':', alpha=0.6)
            if idx == 0: plt.legend()
            
        plt.suptitle(f"Lax Problem Check: {p_str}", fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(IMG_DIR, f"check_{p_str}.png")
        plt.savefig(save_path)
        print(f"   -> Saved: {save_path}")

    print("\n✅ Verification Done. Please check the images!")

if __name__ == "__main__":
    check_and_plot()