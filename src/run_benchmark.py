# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 22:15:39 2026

@author: wding64
"""

import os
import subprocess
import time
from pathlib import Path

# Get the absolute directory of the current Python script
BASE_DIR = Path(__file__).parent

# --- Configuration ---
# Use pathlib to join paths and convert to string for subprocess and Fortran
SOLVER_EXEC = str(BASE_DIR / "weno_gpu_solver")
# Point INPUT_FILE to the DATA_VALIDATION directory
INPUT_FILE  = str(BASE_DIR / "DATA_VALIDATION" / "input_config3_std.nml")
OUTPUT_FILE = str(BASE_DIR / "flow_standard_config3.dat")

def run_single_benchmark():
    # 1. Check required files
    if not os.path.exists(SOLVER_EXEC):
        print(f"[Error] Solver not found: {SOLVER_EXEC}")
        print("Please compile the Fortran code first (e.g., nvfortran Weno_GPU_Param.f90 -o weno_gpu_solver)")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"[Error] Input file not found: {INPUT_FILE}")
        print("Please ensure the input_config3_std.nml file exists in the DATA_VALIDATION directory!")
        return

    # 2. Clean up old output files (to prevent false positives)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f">>> Old result cleaned up: {OUTPUT_FILE}")

    # 3. Run the solver
    print(f">>> Running standard Config 3 benchmark...")
    print(f"    Input : {INPUT_FILE}")
    print(f"    Output: {OUTPUT_FILE}")
    print("-" * 40)

    start_t = time.time()
    
    # Construct the command: ./solver input_file output_file
    cmd = [SOLVER_EXEC, INPUT_FILE, OUTPUT_FILE]
    
    try:
        # text=True handles string outputs natively, and omitting capture_output 
        # allows the solver's real-time output (e.g., time steps) to print to the screen
        subprocess.run(cmd, check=True, text=True)
        
        dur = time.time() - start_t
        print("-" * 40)
        
        if os.path.exists(OUTPUT_FILE):
            print(f">>> [SUCCESS] Run completed! Time elapsed: {dur:.2f}s")
            print(f">>> Results saved to: {OUTPUT_FILE}")
            print(">>> Now please run the visualization script to check if the flow field has rolled up.")
        else:
            print(">>> [WARNING] Solver finished running, but no output file was generated.")

    except subprocess.CalledProcessError as e:
        print(f"\n>>> [FAILED] Solver crashed with error code: {e.returncode}")

if __name__ == "__main__":
    run_single_benchmark()