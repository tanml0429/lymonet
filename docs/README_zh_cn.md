
#### [English](https://github.com/tanml0429/LymoNet) | 简体中文

如果本项目对你有帮助，请点击项目右上角star支持一下和引用下面的论文。

# LymoNet

这个项目是论文的《LymoNet: 一种用于超声图像的先进淋巴结检测网络》的代码、数据集和教程。

 LymoNet(淋巴结检测网络)被提出用于从超声图像中检测和分类正常、炎症和转移性淋巴结。

![GA](https://github.com/tanml0429/lymonet/blob/main/docs/GA.jpg)

图形摘要

# Highlights

1. LymoNet是一种基于yolov8的模型，显著提高了超声图像中淋巴结的自动检测性能。

2. 利用先进的注意力机制和医学知识嵌入提高淋巴结分类准确率。

3. LymoNet达到了SOTA，展示了其在临床应用及其他领域的潜力。

# 贡献Contributions

1. 使用C2fCA和C2fMHSA增强的YOLOv8和YOLOv8-cls具有优越的特征提取功能。

2. 将医学知识整合到YOLOv8-cls中，提高分类精度。

3. 开发LymoNet-Fusion，结合改进的模型，以实现最先进的性能。

# 安装Installation
安装LymoNet代码并配置环境，请查看：
[docs/INSTALL.md](https://github.com/tanml0429/LymoNet/blob/master/docs/INSTALL.md)

# 验证集和权重Validation data and trained weights
下载验证集和训练好的权重，请查看：
[docs/DATASETS.md](https://github.com/tanml0429/LymoNet/blob/master/docs/DATASETS.md)

# 快速开始Quick Start
## 训练Train

安装LymoNet代码，配置环境和下载数据集后，输入代码训练：
```
python train.py 
```
训练结果和权重将保存在 runs/detect/xxx 目录中。

主要的可选参数：
```
--model "x.yaml"
--data  "xx.yaml"
--device "0, 1"  # cpu or gpu id
--imgsz 640 
--batch 32 
--epochs 300 
```

## 验证val

```
python val.py
```

主要的可选参数:
```
--model "xx.pt"
--data  "xx.yaml"
--device "0, 1"  # cpu or gpu id
--imgsz 640 
--batch 32
--fine_cls False
```









# 贡献者Contributors
LymoNet的作者是: Menglu Tan, Yaxin Hou, Zhengde Zhang, Guangdong Zhan, Zijin Zeng, Zunduo Zhao, Hanxue Zhao, and Lin Feng

目前，LymoNet由Menglu Tan (tanml0429@gmail.com)负责维护。

如果您有任何问题，请随时与我们联系。



# 致谢Acknowledgement

本项工作得到了以下资助：北京市杰出青年学者基金（项目编号：JQ22022）、国家重点研发计划（项目编号：2022YFF1502000）以及北京市医院管理局临床医学发展专项资金支持（项目编号：ZYLX202104）。

我们非常感谢
[ultralytics](https://github.com/ultralytics/ultralytics)
项目提供的目标检测算法基准。



如果对您有帮助，请为点击项目右上角的star支持一下或引用论文。

# 引用Citation
```
@article{CDNet,
author={Zheng-De Zhang, Meng-Lu Tan, Zhi-Cai Lan, Hai-Chun Liu, Ling Pei and Wen-Xian Yu},
title={CDNet: a real-time and robust crosswalk detection network on Jetson nano based on YOLOv5},
Journal={Neural Computing and Applications}, 
Year={2022},
DOI={10.1007/s00521-022-07007-9},
}
```


# 许可License
LymoNet可免费用于非商业用途，并可在这些条件下重新分发。 如需商业咨询，请发送电子邮件至tanml0429@gmail.com，我们会将详细协议发送给您。