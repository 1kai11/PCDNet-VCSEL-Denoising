# PCDNet-VCSEL-Denoising
Official PyTorch implementation of PCDNet for VCSEL defect image denoising (The Visual Computer submission)
## Overview
Vertical-cavity surface-emitting laser (VCSEL) defect images acquired via infrared electroluminescence inspection are often severely degraded by mixed industrial noise, which obscures defect boundaries and annular structures.  
To address this problem, we propose **PCDNet**, a lightweight denoising network that explicitly incorporates polar-coordinate priors to better preserve annular textures and structural continuity.

This repository provides the complete implementation of the proposed network, including model definition, loss functions, and training scripts, to facilitate reproducibility and further research.
## Repository Structure
PCDNet-VCSEL-Denoising/
├── model.py # PCDNet network architecture
├── loss.py # Hybrid loss functions
├── train.py # Training script
├── plug_play/ # Plug-and-play modules for integration into other denoisers
├── requirements.txt # Python dependencies
└── README.md
## Environment
The code has been tested under the following environment:

- Python 3.9  
- PyTorch 1.10.1  
- CUDA 11.3  
- NVIDIA GPU

Other PyTorch versions may also work but have not been extensively tested.
