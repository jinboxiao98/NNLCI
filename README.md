# NNLCI: Neural Network with Local Converging 

**NNLCI** is a deep learning framework for high-fidelity flow field reconstruction in fluid dynamics. It utilizes a locally converging input (stencil) approach to map dual low-fidelity (LF) coarse grids to a high-fidelity (HF) fine grid

---

## How to get started?

### 0. Install Drivers & Environment

To run the NNLCI framework, you must prepare the environment for both high-performance CFD simulation and deep learning.

<details>
<summary><strong>0.1 Install NVIDIA GPU Drivers</strong></summary>
<br>

The NVIDIA Driver is the foundation for both the CUDA-accelerated Fortran solver and PyTorch training.

* **Windows:** * Download the latest **Game Ready Driver** or **Studio Driver** [Official NVIDIA Website](https://www.nvidia.com/download/index.aspx).
    * Run the `.exe` and follow the installation prompts.
    * Restart your computer.
* **Linux (Ubuntu):** 
    * Use `sudo apt install nvidia-driver-535` to install NVIDIA driver.
    * Reboot to load the driver `sudo reboot`.
    * Verify installation `nvidia-smi`.

</details>

<details>
<summary><strong>0.2 Install NVIDIA HPC SDK (for nvfortran)</strong></summary>
<br>

The CFD solver (`Weno_GPU_Charac_Final.f90`) requires the **NVIDIA HPC SDK**, which provides the `nvfortran` compiler used to build the GPU-accelerated binary.

1.  Visit the [NVIDIA HPC SDK Download Page](https://developer.nvidia.com/hpc-sdk).
2.  **For Linux (Recommended):** Follow the installation instructions for your specific distribution.
3.  **Set Environment Variables:** Add the following to your `~/.bashrc`:
    ```bash
    export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin:$PATH
    export MANPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/man:$MANPATH
    ```
</details>

<details>
<summary><strong>0.3 Install Python Dependencies</strong></summary>
<br>

The NNLCI pipeline (preprocessing, training, and post-processing) relies on several Python libraries.

##### Option 1: Using pip (Recommended for Virtual Environments)
1.  **Install PyTorch with CUDA support** (Replace `cu121` with your matching CUDA version):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
2.  **Install General Dependencies:**
    ```bash
    pip install numpy scikit-image tqdm matplotlib
    ```

##### Option 2: Using conda

1.  **Create a new environment**
    ```bash
    conda create -n nnlci_env python=3.10
    conda activate nnlci_env
    ```

2.  **Install PyTorch and other libraries**
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install numpy scikit-image tqdm
    ```
</details>

<details>
<summary><strong>0.4 Verification</strong></summary>
<br>

To verify the installation, run the following commands:
* **Verify Fortran Compiler:**
    ```bash
    nvfortran --version
    ```

* **Verify PyTorch GPU Access:**
    ```bash
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0)}')"
    ```

</details>


### 1. Data Generation: The CFD Solver
The training data is generated using a high-order **WENO-5 JS** scheme solver optimized for GPUs.

#### Step 1.1: Compile the Solver
1.  Navigate to the `src/` directory.
2.  Compile the solver:
    ```bash
    nvfortran Weno_GPU_Charac_Final.f90 -o weno_gpu_solver
    ```
    The solver relies on `gpu_data.mod` and `kernels.mod` for dynamic memory allocation and computation kernels. 
    
    *Note: The executable must be named `weno_gpu_solver` for the Python automation scripts to function correctly.*

NNLCI uses standard **2D Riemann Problems** (Lax & Liu configurations) to learn flow physics.

#### Step 1.2: Generate Configuration Files
Run the setup script to create the `.nml` namelist files for various configurations (e.g., Config 3, 4, and 6).
```bash
python generate_standard_cases.py
```

#### Step 1.3: Execute the CFD Solver
The data generation process is independent for each configuration. To generate data for a specific case (e.g., Config 3):
1.  **Navigate** to the `DATA_VALIDATION/` directory.
2.  **Modify the Configuration:** Open `input_standard_config3.nml` and systematically perturb the initial conditions (e.g., pressure, density, or velocity) by specific increments such as $\pm5\%$, $\pm10\%$, etc.
3.  **Execute the Solver:**
    ```bash
    python run_benchmark.py
    ```
4.  **Iterate for Diversity:** Repeat the process by applying different perturbation percentages until a comprehensive range of CFD results (solution space) is covered.

This produces the high-fidelity (800x) and low-fidelity (100x, 200x) datasets needed for the pipeline.


### 2. The NNLCI Pipeline
#### Step 2.1: Preprocessing & Smart Sampling
The `NNLCI_preprocessing.py` script prepares the data using a **Smart Sampling (Shock Masking)** strategy.

* **Upsampling**: It uses bilinear interpolation to align 100x and 200x grids to the 800x target.

* **Shock Masking**: To capture critical physics, the script calculates gradients to identify shock waves, keeping **40% of samples from shock regions** and **10% from smooth regions**.

* **Stencil Extraction**: It extracts a 24-point spatial stencil around each coordinate, creating a **72-dimensional input vector**.

```bash
python NNLCI_preprocessing.py
```

#### Step 2.2: Training the Deep MLP
Run `NNLCI_training.py` to train the Deep MLP model (`NNLCI_Net`).
* **Architecture**: A deep MLP with **10 hidden layers** (600 neurons each) using **Tanh** activation.

* **Efficiency**: The script uses **Memory Mapping (`mmap`)** to handle massive datasets that exceed system RAM.

* **Hyperparameters**: Default batch size is **65536** with an Adam optimizer (Initial LR: `1e-4`).
```bash
python NNLCI_training.py
```

#### Step 2.3: Evaluation & Metrics
Run `NNLCI_post_processing.py` to evaluate the trained model on unseen test data

* **Metrics**: The script calculates **SSIM** (Structural Similarity Index) and **PSNR** (Peak Signal-to-Noise Ratio) to quantify reconstruction quality
```bash
python NNLCI_post_processing.py
```



### 3. Further NNLCI Extensions
**NNLCI** is designed as a modular framework, allowing users to replace the provided WENO-5 JS solver with their own CFD tools, including commercial software such as **Ansys Fluent**, **OpenFOAM**, or **Star-CCM+**.

#### 3.1. Data Preparation Requirements
NNLCI is a supervised deep learning framework. To apply NNLCI to a specific problem, you must prepare multiple training datasets consisting of synchronized flow fields at different resolutions. To improve model robustness, it is highly recommended to **perturb initial conditions** (e.g., by $\pm5\%$ or $\pm10\%$) or vary geometries to create a diverse range of CFD results.

For each case, you need three synchronized grids:

1. **High-Fidelity (HF) Grid**: The "Ground Truth" resolution target.
2. **Low-Fidelity 1 (LF1) Grid**: The first coarse resolution input.
3. **Low-Fidelity 2 (LF2) Grid**: The second coarse resolution input.

#### 3.2 Extract Stencils
The core of NNLCI training is the mapping of local spatial information to a high-resolution output point.

- **The Training Pair**: A single training sample consists of a **72-dimensional input vector**(derived from dual LF stencils) and a **4-dimensional output vector** (the refined HF center point) for 2D Euler equation cases.

- **Stencil Composition**: Each input stencil is composed of 24 surrounding points plus one center point (typically forming a 5x5 spatial neighborhood).

- **Coordinate Alignment**: Because the HF and LF grids have different densities, **bilinear interpolation** (implemented via `pytorch_upsample` in our code) is required to align the coarse stencils to the fine grid coordinates.

- **Implementation**: In the provided pipeline, we use the `extract_patches_masked` function to generate these pairs from raw CFD data.

**Normalization**: Physical variables (Velocity, Pressure, Density) must be scaled into the \([-1, 1]\) range using the `MinMaxScalerMinus1To1` class 12. This range is optimal for the **Tanh** activation functions used in the 10-layer `NNLCI_Net`

#### 3.3 Advanced Sampling
While every point in a high-resolution grid could theoretically be a training sample, processing every coordinate would quickly **exceed GPU memory** and system RAM.

**Smart Sampling** is necessary to extract only the points that reflect the most critical flow field structures. For high-speed flows involving discontinuities:

- **Shock Masking**: Use the `compute_gradient_mask` logic to identify and prioritize high-gradient regions.

- **Attention Balancing**: Adjust the `SHOCK_RATIO` (default 0.4) and `SMOOTH_RATIO` (default 0.1) to balance the network's learning between complex shock waves and smooth background flow.

- **Efficiency**: This strategy discards redundant smooth-flow data (typically the remaining 50% of the field), allowing for much larger training batch sizes (e.g., 65,536) and faster convergence.


---
## Citation

If you use **NNLCI** in your research, please consider citing it as follows. **Ref. 1 and 2** provide the foundational theory and methodology for Neural Networks with Local Converging Inputs (NNLCI), originally proposing the framework for solving 1D and 2D conservation laws. If your work involves complex geometries or flexible grid configurations, please cite **ref. 3**, which demonstrates the method's extension and flexibility in unstructured-grid computational fluid dynamics.

<p align="center">
  <a href="https://arc.aiaa.org/doi/full/10.2514/1.J063885">
    <img src="https://img.shields.io/badge/DOI-10.2514%2F1.J063885-9C92AC.svg" alt="DOI 1">
  </a>
  <a href="https://doi.org/10.4208/cicp.OA-2022-0285">
    <img src="https://img.shields.io/badge/DOI-10.4208%2Fcicp.OA--2022--0285-E67E22.svg" alt="DOI 2">
  </a>
  <a href="https://doi.org/10.4208/cicp.OA-2023-0026">
    <img src="https://img.shields.io/badge/DOI-10.4208%2Fcicp.OA--2023--0026-3174A1.svg" alt="DOI 3">
  </a>
</p>

```bibtex
@article{huang2023neural,
  title={Neural networks with local converging inputs (NNLCI) for solving conservation laws, Part I: 1D problems},
  author={Huang, Haoxiang and Yang, Vigor and Liu, Yingjie},
  journal={Communications in Computational Physics},
  volume={34},
  number={2},
  pages={290--317},
  year={2023},
  publisher={Global Science Press}
}

@article{huang2023neural,
  title={Neural networks with local converging inputs (NNLCI) for solving conservation laws, Part II: 2D problems},
  author={Huang, Haoxiang and Yang, Vigor and Liu, Yingjie},
  journal={Communications in Computational Physics},
  volume={34},
  number={4},
  pages={907--933},
  year={2023},
  publisher={Global Science Press}
}

@article{ding2024neural,
  title={Neural network with local converging input for unstructured-grid computational fluid dynamics},
  author={Ding, Weiming and Huang, Haoxiang and Lee, Tzu-Jung and Liu, Yingjie and Yang, Vigor},
  journal={AIAA Journal},
  volume={62},
  number={8},
  pages={3155--3166},
  year={2024},
  publisher={American Institute of Aeronautics and Astronautics}
}
```
