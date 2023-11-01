# ScratchFormer

# Remote Sensing Change Detection With Transformers Trained from Scratch
This repo contains the official **PyTorch** code for Remote Sensing Change Detection With Transformers Trained from Scratch [[Arxiv]](https://arxiv.org/pdf/2304.06710.pdf). 

**Code is released!**

Highlights
-----------------
- **Trained From Scratch:** Our proposed solution for remote sensing change detection (CD) is called ScratchFormer, which utilizes a transformers-based Siamese architecture. Notably, ScratchFormer does not depend on pretrained weights or the need to train on another CD dataset.
change detection (CD).
- **Shuffled Sparse Attention:** The proposed ScratchFormer model incorporates a novel operation called shuffled sparse attention (SSA), which aims to improve the model's ability to focus on sparse informative regions that are important for the remote sensing change detection (CD) task.
- **Change-Enhanced Feature Fusion:** In addition, we present a change-enhanced feature fusion module (CEFF) that utilizes per-channel re-calibration to improve the relevant features for semantic changes, while reducing the impact of noisy features.

Methods
-----------------
<img width="1096" alt="image" src="https://github.com/mustansarfiaz/ScratchFormer/blob/main/demo/proposed_framework.jpg">

