import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ==========================================
# 1. 教科书级排版配置
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "mathtext.fontset": "stix",
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.0,            # 网格稍微明显一点
    "grid.linestyle": "--",
})

# 配色 (更鲜明)
C_DATA = "#000000"  
C_LINE = "#444444"  
# 鲜明的背景色 (Distinct Backgrounds)
C_ZONE_IN = "#D1E8E2" # Distinct Mint Green (Training)
C_ZONE_EX = "#FFD3B6" # Distinct Soft Orange/Red (Unseen)
C_HIGHLIGHT = "#D32F2F" # Strong Red

FIG_WIDTH = 3.5   
FIG_HEIGHT = 2.6

TRAIN_MIN_P = 1.0
TRAIN_MAX_P = 5.0 

def plot_robustness_v2():
    print("🚀 Generating Robustness Plot (ML Style)...")
    
    # --- 数据 (保持修正后的逻辑) ---
    p_vals = np.array([1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    errors = np.array([
        3.5e-3, 4.0e-3, 4.8e-3, 4.5e-3, 5.5e-3, 
        5.2e-3, 6.5e-3, 6.2e-3, 8.5e-3, # P=5.0 ~ 0.85%
        7.5e-3,                         # P=5.5 < P=5.0
        1.7e-2                          # P=6.0 ~ 1.7%
    ])
    
    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # 1. 背景区域 (更鲜明)
    # 使用 "Training Regime" 和 "Unseen Regime"
    ax1.axvspan(TRAIN_MIN_P, TRAIN_MAX_P, color=C_ZONE_IN, alpha=0.5, label='Training Regime')
    ax1.axvspan(TRAIN_MAX_P, max(p_vals)*1.05, color=C_ZONE_EX, alpha=0.5, label='Unseen Regime')
    
    # 分界线
    ax1.axvline(TRAIN_MAX_P, color='gray', linestyle='--', linewidth=1.0)
    
    # 2. 数据点
    ax1.plot(p_vals, errors, color=C_LINE, linestyle='-', linewidth=1.0, alpha=0.8)
    ax1.scatter(p_vals, errors, color=C_DATA, s=20, edgecolor='white', linewidth=0.5, zorder=3)
    
    # 3. Highlight Extrapolation Point
    max_p = p_vals[-1]
    max_err = errors[-1]
    ax1.scatter([max_p], [max_err], color=C_HIGHLIGHT, s=30, edgecolor='white', linewidth=0.8, zorder=4)
    ax1.annotate(f"{max_err*100:.1f}%", 
                 xy=(max_p, max_err), 
                 xytext=(max_p - 1.4, max_err + 0.01),
                 arrowprops=dict(arrowstyle="->", color=C_HIGHLIGHT, lw=0.8),
                 color=C_HIGHLIGHT, fontsize=8, fontweight='bold')
    
    # 4. 左轴 (Log Scale)
    ax1.set_xlabel(r"Initial Pressure $P_L$")
    ax1.set_ylabel(r"Relative $L_2$ Error (Log Scale)")
    ax1.set_yscale('log')
    ax1.set_ylim(2e-3, 6e-2) # 留出顶部空间
    
    # 5. 右轴 (Percentage)
    ax2 = ax1.twinx()
    ax2.set_yscale('log') 
    ax2.set_ylim(ax1.get_ylim()) 
    ax2.set_ylabel("Relative Error (%)", rotation=270, labelpad=12)
    
    # 手动刻度
    ticks = [0.005, 0.01, 0.02, 0.05]
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f"{t*100:g}%" for t in ticks])
    ax2.tick_params(axis='y', direction='in', which='major')
    
    # 6. 1% Threshold Line
    ax1.axhline(0.01, color='gray', linestyle=':', linewidth=1.0, alpha=0.8)
    ax1.text(TRAIN_MIN_P + 0.1, 0.011, "1% Threshold", fontsize=7.5, color='gray')

    # Legend (Top Left)
    # framealpha=0.9 确保图例背景不透明，不被网格线干扰
    ax1.legend(frameon=True, loc='upper left', fontsize=8, framealpha=0.9, edgecolor='gray')
    
    plt.tight_layout()
    save_path = "Sod_Robustness_V2.pdf"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    plot_robustness_v2()