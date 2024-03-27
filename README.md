[![Stars](https://img.shields.io/github/stars/tanml0429/LNDNet)](
https://github.com/tanml0429/LNDNet)
[![Open issue](https://img.shields.io/github/issues/tanml0429/LNDNet)](
https://github.com/tanml0429/LNDNet/issues)
[![Datasets](https://img.shields.io/static/v1?label=Download&message=source_code&color=orange)](
https://github.com/tanml0429/LNDNet/archive/refs/heads/main.zip)

#### English | [简体中文](https://github.com/tanml0429/LNDNet/blob/main/docs/README_zh_cn.md)

Please **star this project** in the upper right corner and **cite this paper** blow 
if this project helps you. 

# LymoNet

This repository is the source codes for the paper 
"LymoNet: An Advanced Lymph Node Detection Network for Ultrasound Images".

LymoNet (Lymph Node Detection Network) is proposed to detect and classify normal, inflammatory, and metastatic lymph nodes from ultrasound images.


![GA](GA.jpg)
Graphical abstract.

# Highlights
1. LymoNet, a YOLOv8-based model, significantly improving automated lymph node detection from ultrasound images.

2. Utilization of advanced attention mechanisms and medical knowledge embedding to enhance classification accuracy of lymph nodes.

3. LymoNet achieves state-of-the-art performance, demonstrating its potential in clinical applications and beyond.

# Contributions

1. Enhanced YOLOv8 and YOLOv8-cls with C2fCA and C2fMHSA for superior feature extraction.

2. Incorporated medical knowledge into YOLOv8-cls for refined classification accuracy.

3. Developed LymoNet-Fusion, combining improved models to achieve state-of-the-art performance.

# Installation
Get CDNet code and configure the environment, please check out [docs/INSTALL.md](https://github.com/tanml0429/LymoNet/blob/master/docs/INSTALL.md)

# Datasets and trained weights
Download datasets and trained weights, please check out [docs/DATASETS.md](https://github.com/tanml0429/LymoNet/blob/master/docs/DATASETS.md)

# Quick Start
## Train

Once you get the LymoNet code, configure the environment and download the dataset, just type:
```
python train.py 
```
The training results and weights will be saved in runs/detect/directory.

The main optional arguments:
```
--model "x.yaml"
--data  "xx.yaml"
--device "0, 1"  # cpu or gpu id
--imgsz 640 
--batch 32 
--epochs 300 
```


## Val
```
python val.py
```
The main optional arguments:
```
--model "xx.pt"
--data  "xx.yaml"
--device "0, 1"  # cpu or gpu id
--imgsz 640 
--batch 32
--fine_cls False
```

# Contributors

LymoNet is authored by Menglu Tan, Yaxin Hou, Zhengde Zhang, Guangdong Zhan, Zijin Zeng, Zunduo Zhao, Hanxue Zhao, and Lin Feng.

Currently, it is maintained by Menglu Tan (tanml0429@gmail.com).

# Acknowledgement

This work was supported by the Beijing Municipal Fund for Distinguished Young Scholars (Grand No. JQ22022), National Key R&D Program of China (Grant No. 2022YFF1502000), and the Beijing Hospitals Authority Clinical Medicine Development of Special Funding Support (Grant No. ZYLX202104). 

We are very grateful to the [ultralytics](https://github.com/ultralytics/ultralytics) project for the benchmark detection algorithm.



# Citation
```

```


# License
LymoNet is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at tanml0429@gmail.com. We will send the detail agreement to you.










