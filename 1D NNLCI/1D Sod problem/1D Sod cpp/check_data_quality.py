# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 22:56:25 2026

@author: wding64
"""

import os
import glob
import pandas as pd
import numpy as np

# --- 配置 ---
DATA_DIR = "./data"
# 阈值：认为小于此值的密度/压强是由于 Solver 崩溃引起的
CRITICAL_NEGATIVE_THRESHOLD = -0.5 
# 阈值：认为大于此值的数据是 Solver 爆炸
EXPLOSION_THRESHOLD = 1e6

def check_files():
    print(f"🔍 Scanning files in {DATA_DIR}...")
    files = glob.glob(os.path.join(DATA_DIR, "*.dat"))
    
    if not files:
        print("❌ No .dat files found!")
        return

    bad_files = []
    suspicious_files = []
    
    for f in sorted(files):
        filename = os.path.basename(f)
        try:
            # 读取数据
            df = pd.read_csv(f, sep='\s+')
            
            # 1. 检查 NaN 或 Inf
            if df.isnull().values.any():
                print(f"❌ [NAN FOUND] {filename} contains NaNs!")
                bad_files.append(filename)
                continue
            
            if np.isinf(df.values).any():
                print(f"❌ [INF FOUND] {filename} contains Infinite values!")
                bad_files.append(filename)
                continue

            # 2. 检查数值范围 (Solver Explosion)
            max_val = df.max().max()
            min_val = df.min().min()
            
            if max_val > EXPLOSION_THRESHOLD:
                print(f"💥 [EXPLOSION] {filename}: Max value {max_val:.2e} exceeds threshold!")
                bad_files.append(filename)
                continue
            
            # 3. 检查物理非负性 (Density & Pressure)
            # 注意：高阶格式可能有轻微负振荡 (e.g. -1e-3)，这是允许的，能被 Log Shift 处理
            # 但如果出现 -10, -50 这种，就是 Solver 错了。
            min_den = df['den'].min()
            min_pres = df['pres'].min()
            
            if min_den < CRITICAL_NEGATIVE_THRESHOLD or min_pres < CRITICAL_NEGATIVE_THRESHOLD:
                print(f"⚠️ [SEVERE NEGATIVE] {filename}: Min Den={min_den:.4f}, Min Pres={min_pres:.4f}")
                bad_files.append(filename)
            elif min_den < 0 or min_pres < 0:
                # 记录轻微负值但不报错，仅供参考
                # print(f"ℹ️ [Minor Neg] {filename}: {min_val:.4f} (Likely oscillation)")
                suspicious_files.append((filename, min_val))

        except Exception as e:
            print(f"❌ [READ ERROR] {filename}: {e}")
            bad_files.append(filename)

    print("\n" + "="*40)
    print("📊 Scan Report")
    print("="*40)
    
    if bad_files:
        print(f"🚫 Found {len(bad_files)} CORRUPTED files (DO NOT TRAIN ON THESE):")
        for f in bad_files:
            print(f"  - {f}")
        print("\n💡 Recommendation: Identify the grid/pressure parameters of these files.")
        print("   If they are Low-Fi cases, your CFL might be too high for the 1st order solver.")
        print("   If they are High-Fi cases, the limiter might be failing at extreme pressures.")
    else:
        print("✅ No corrupted files found (NaN/Inf/Explosion).")
        
    if suspicious_files:
        worst_oscillation = min([v for k, v in suspicious_files])
        print(f"\nℹ️ Found {len(suspicious_files)} files with minor negative oscillations.")
        print(f"   Worst value: {worst_oscillation:.5f}")
        print("   (These are usually handled by the Log Shift, assuming Shift > |Worst Value|)")

if __name__ == "__main__":
    check_files()