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

实验结果汇总
指标
模型	
Iou_1	
F1_1

Initial(初始)	
84.68	
91.71

Best	
85.06	
91.93

+ESAM(only stage1)in2p	
83.93	
91.26

+ESAM(stage1 & stage2)in2p	
暂无	

+ESAM(stage1 & stage2) in1p	
暂无	

+CBAM(all stages)	
84.36	
		
		
		
		
		
		
		
		
		
		
		
