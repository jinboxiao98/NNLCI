# NNLCI: Neural Network with Local Converging 

**NNLCI** is a deep learning framework for high-fidelity flow field reconstruction in fluid dynamics. It utilizes a locally converging input (stencil) approach to map dual low-fidelity (LF) coarse grids to a high-fidelity (HF) fine grid



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