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

<!DOCTYPE html>
<html>
<head>
    <title>实验结果汇总</title>
</head>
<body>
    <table border="1">
        <tr>
            <th></th>
            <th>iou_1</th>
            <th>f1_1</th>
        </tr>
        <tr>
            <td>initial(初始)</td>
            <td>84.68</td>
            <td>91.71</td>
        </tr>
        <tr>
            <td>best performence</td>
            <td>85.06</td>
            <td>91.93</td>
        </tr>
        <tr>
            <td>+ESAM(only stage1)in2p</td>
            <td>83.93</td>
            <td>91.26</td>
        </tr>
        <tr>
            <td>+ESAM(stage1 & stage2)in2p</td>
            <td>暂无</td>
            <td>暂无</td>
        </tr>
        <tr>
            <td>+ESAM(stage1 & stage2) in1p</td>
            <td>暂无</td>
            <td>暂无</td>
        </tr>
        <tr>
            <td>+CBAM(all stages)</td>
            <td>84.36</td>
            <td></td>
        </tr>
        <tr>
            <td>+CrossAtt(all stage)</td>
            <td>暂无</td>
            <td>暂无</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
        </tr>
    </table>
</body>
</html>

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
