# PCDNet-VCSEL-Denoising
Official PyTorch implementation of PCDNet for VCSEL defect image denoising (The Visual Computer submission)

## Overview
Vertical-cavity surface-emitting laser (VCSEL) defect images acquired via infrared electroluminescence inspection are often severely degraded by mixed industrial noise, which obscures defect boundaries and annular structures.  
To address this problem, we propose **PCDNet**, a lightweight denoising network that explicitly incorporates polar-coordinate priors to better preserve annular textures and structural continuity.

## Core Method: Polar-Coordinate Constrained Deformable Convolution (Polar-DCN)
VCSEL defect images exhibit strong annular and rotationally symmetric structures, which are difficult to model effectively using conventional convolution or unconstrained deformable convolution.
To explicitly exploit this geometric prior, PCDNet introduces a Polar-Coordinate Constrained Deformable Convolution (Polar-DCN) module.

Polar-DCN is built upon standard deformable convolution but incorporates directional constraints in the polar coordinate system. Instead of learning unrestricted 2D offsets, the sampling offsets are decomposed into radial and tangential components, corresponding to directions perpendicular and parallel to the annular structure. This design enforces angular continuity and aligns adaptive sampling with the intrinsic ring-shaped geometry of VCSEL defect regions.

By constraining offset learning to orthogonal radial–tangential directions, Polar-DCN improves feature sampling along circular edges, enhances annular texture preservation, and stabilizes offset optimization by avoiding uncontrolled deformation.
The module is embedded within a Polar Coordinate Residual Block (PCRB) and further enhanced with efficient channel attention to balance structural fidelity and noise suppression.

This geometry-aware design enables PCDNet to achieve effective denoising under mixed-noise conditions while maintaining a lightweight architecture, and it can be flexibly integrated into other denoising frameworks in a plug-and-play manner.

## Repository Structure
This repository provides the complete implementation of the proposed network, including model definition, loss functions, and training scripts, to facilitate reproducibility and further research.
PCDNet-VCSEL-Denoising/
  - model.py # PCDNet network architecture
  - loss.py # Hybrid loss functions
  - train.py # Training script
  - plug_play/ # Plug-and-play modules for integration into other denoisers
  - requirements.txt # Python dependencies
  - README.md

## Environment
The code has been tested under the following environment:

- Python 3.9  
- PyTorch 1.10.1  
- CUDA 11.3  
- NVIDIA GPU

Other PyTorch versions may also work but have not been extensively tested.
