import os
import re
import time
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import glob 

# ==============================================================================
# Configuration Area
# ==============================================================================
MY_CODE_DIR = "."
HIS_CODE_DIR = "./2D Riemann WENO3"

TARGET_NX = 800
TARGET_NY = 800
TARGET_T  = 0.25

# Compilation commands (unchanged)
CMD_COMPILE_MY  = "nvfortran -O3 -cuda -gpu=cc80 Weno_GPU_Charac_Final.f90 -o weno5_solver"
CMD_COMPILE_HIS = "nvcc -O3 main.cu -o weno3_solver" 

# [Core Modification] Profiling command prefix
# --stats=true: Print statistical tables after completion
# -f: Force overwrite of old reports
PROFILE_CMD = "nsys profile --stats=true -f true"

# ==============================================================================
# 1. Preparation: Patch His Code (unchanged)
# ==============================================================================
def patch_his_code():
    print(f">>> [Setup] Patching his code in {HIS_CODE_DIR}...")
    main_cu_path = os.path.join(HIS_CODE_DIR, "main.cu")
    if not os.path.exists(main_cu_path): return False
    with open(main_cu_path, 'r') as f: content = f.read()
    content = re.sub(r'#define\s+IIX\s+\d+', f'#define IIX {TARGET_NX}', content)
    content = re.sub(r'#define\s+IIY\s+\d+', f'#define IIY {TARGET_NY}', content)
    content = re.sub(r'#define\s+TOUT\s+[\d\.]+', f'#define TOUT {TARGET_T}', content)
    with open(main_cu_path, 'w') as f: f.write(content)
    return True

# ==============================================================================
# 2. Preparation: Generate My Input (unchanged)
# ==============================================================================
def generate_my_input():
    print(f">>> [Setup] Generating input_battle.nml...")
    nml_content = f"""
&CASE_PARAMS
  q1_r=1.5, q1_u=0.0, q1_v=0.0, q1_p=1.5,
  q2_r=0.5323, q2_u=1.206, q2_v=0.0, q2_p=0.3,
  q3_r=0.138, q3_u=1.206, q3_v=1.206, q3_p=0.029,
  q4_r=0.5323, q4_u=0.0, q4_v=1.206, q4_p=0.3,
  DT=0.0001,
  T_END={TARGET_T},
  NX={TARGET_NX}, NY={TARGET_NY}
/
"""
    with open("input_battle.nml", "w") as f: f.write(nml_content)

# ==============================================================================
# 3. Run Function with Profiling
# ==============================================================================
def run_profiled_benchmark():
    # --- Compile ---
    print("\n>>> [Compiling]...")
    subprocess.run(CMD_COMPILE_MY, shell=True, check=True)
    subprocess.run(f"cd '{HIS_CODE_DIR}' && {CMD_COMPILE_HIS}", shell=True, check=True)

    # --- Run My Code with Profiler ---
    print("\n" + "="*60)
    print(">>> [PROFILING] MY AI CODE (WENO5)")
    print("="*60)
    if os.path.exists("flow_battle.dat"): os.remove("flow_battle.dat")
    
    # Construct command: nsys profile ... ./solver input.nml
    my_cmd = f"{PROFILE_CMD} -o profile_my ./weno5_solver input_battle.nml flow_battle.dat"
    
    start_t = time.time()
    # check=False allows us to capture output even if nsys returns non-zero 
    # (some profiling warnings can cause non-zero returns)
    proc_my = subprocess.run(my_cmd, shell=True, capture_output=True, text=True)
    my_time = time.time() - start_t
    
    # Print the statistical output from nsys
    print_nsys_summary(proc_my.stdout, "MY CODE")

    # --- Run His Code with Profiler ---
    print("\n" + "="*60)
    print(">>> [PROFILING] EXPERT CODE (WENO3)")
    print("="*60)
    subprocess.run(f"rm -f '{HIS_CODE_DIR}'/RESU*.DAT", shell=True)
    
    his_cmd = f"cd '{HIS_CODE_DIR}' && {PROFILE_CMD} -o profile_his ./weno3_solver"
    
    start_t = time.time()
    proc_his = subprocess.run(his_cmd, shell=True, capture_output=True, text=True)
    his_time = time.time() - start_t
    
    print_nsys_summary(proc_his.stdout, "EXPERT CODE")
    
    return my_time, his_time

def print_nsys_summary(nsys_output, label):
    """Extract key tables from nsys output"""
    print(f"\n--- {label} GPU STATISTICS ---")
    
    lines = nsys_output.splitlines()
    printing = False
    found_stats = False
    
    for line in lines:
        # Look for the start of the statistical tables
        if "CUDA Kernel Statistics" in line or "CUDA Memory Operation Statistics" in line:
            printing = True
            found_stats = True
            print(f"\n>> {line.strip()}")
            continue
        
        if printing:
            if line.strip() == "": # An empty line usually means the end of the table
                printing = False
                continue
            # Print table content
            print(line)
            
    if not found_stats:
        print("[Warning] Could not parse nsys stats. Raw output tail:")
        print("\n".join(lines[-20:]))

# ==============================================================================
# 4. Plotting (unchanged)
# ==============================================================================
def load_tecplot_slice(filepath, target_y_idx):
    try:
        with open(filepath, 'r') as f: lines = f.readlines()
        start_line = 0
        for i, line in enumerate(lines):
            if "ZONE" in line: 
                start_line = i + 1; break
        
        if len(lines) == 0: return None, None
        
        raw_str = "".join(lines[start_line:])
        fixed_str = re.sub(r'(\d)([+-]\d{3})', r'\1E\2', raw_str)
        data = np.loadtxt(io.StringIO(fixed_str))
        full_grid = data.reshape((TARGET_NY+1, TARGET_NX+1, 6))
        slice_data = full_grid[target_y_idx, :, :]
        return slice_data[:, 0], slice_data[:, 2]
    except: return None, None

def visualize_results(my_time, his_time):
    # Load data (My Code)
    my_x, my_rho = load_tecplot_slice("flow_battle.dat", TARGET_NY // 2)
    
    # Load data (His Code)
    his_files = sorted(glob.glob(os.path.join(HIS_CODE_DIR, "RESU*.DAT")))
    if not his_files: return
    his_x, his_rho = load_tecplot_slice(his_files[-1], TARGET_NY // 2)

    if my_x is None or his_x is None: return

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(his_x, his_rho, 'b--', linewidth=1.5, label=f'Expert (WENO3, C++)\nTime: {his_time:.2f}s')
    plt.plot(my_x, my_rho, 'r-', linewidth=1.5, alpha=0.8, label=f'AI-Aided (WENO5, Fortran)\nTime: {my_time:.2f}s')
    plt.title(f"Benchmark: 2D Riemann Config 3 (Grid: {TARGET_NX}x{TARGET_NX})", fontsize=14)
    plt.xlabel("X (at Y=0.5)")
    plt.ylabel("Density")
    plt.legend(fontsize=12); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("benchmark_result_profiled.png")
    
    print(f"\n{'='*40}\nFINAL SCOREBOARD (Profiled Run)\n{'='*40}")
    print(f"Expert Code: {his_time:.4f} s")
    print(f"Your Code:   {my_time:.4f} s")
    print(f"Speedup:     {(his_time/my_time):.2f}x FASTER")
    print(f"{'='*40}")

if __name__ == "__main__":
    if patch_his_code():
        generate_my_input()
        # Run benchmark with nsys
        t_my, t_his = run_profiled_benchmark()
        visualize_results(t_my, t_his)