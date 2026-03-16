# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 22:15:39 2026

@author: wding64
"""

import os
import subprocess
import time

# --- 配置部分 ---
SOLVER_EXEC = "./weno_gpu_solver"         # 请确认你的可执行文件名
INPUT_FILE  = "input_config3_std.nml" # 刚才创建的标准输入文件
OUTPUT_FILE = "flow_standard_config3.dat"  # 指定输出文件名

def run_single_benchmark():
    # 1. 检查必要文件
    if not os.path.exists(SOLVER_EXEC):
        print(f"[Error] 找不到求解器: {SOLVER_EXEC}")
        print("请先编译 Fortran 代码 (e.g., nvfortran Weno_GPU_Param.f90 -o weno_gpu_solver)")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"[Error] 找不到输入文件: {INPUT_FILE}")
        print("请先创建 input_standard_config6.nml 文件！")
        return

    # 2. 清理旧的输出文件 (防止误判)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f">>> 已清理旧结果: {OUTPUT_FILE}")

    # 3. 运行求解器
    print(f">>> 正在运行标准 Config 3 基准测试...")
    print(f"    Input : {INPUT_FILE}")
    print(f"    Output: {OUTPUT_FILE}")
    print("-" * 40)

    start_t = time.time()
    
    # 构造命令: ./solver input_file output_file
    cmd = [SOLVER_EXEC, INPUT_FILE, OUTPUT_FILE]
    
    try:
        # capture_output=False 让求解器的实时输出直接打印到屏幕上，方便你看到时间步
        subprocess.run(cmd, check=True, text=True)
        
        dur = time.time() - start_t
        print("-" * 40)
        
        if os.path.exists(OUTPUT_FILE):
            print(f">>> [SUCCESS] 运行成功！耗时: {dur:.2f}s")
            print(f">>> 结果已保存至: {OUTPUT_FILE}")
            print(">>> 现在请运行可视化脚本检查流场是否卷起来了。")
        else:
            print(">>> [WARNING] 求解器运行结束，但未生成输出文件。")

    except subprocess.CalledProcessError as e:
        print(f"\n>>> [FAILED] 求解器崩溃，错误码: {e.returncode}")

if __name__ == "__main__":
    run_single_benchmark()