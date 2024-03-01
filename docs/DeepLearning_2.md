---
title: Deep Learning_2
---




### Swin Transformer

[论文](https://arxiv.org/abs/2103.14030)
[作者团队代码](https://github.com/microsoft/Swin-Transformer)

transformer 用作CV领域的骨干网络
patch merging 多尺度的特征图

### DETR

端到端目标检测，基于集合的目标函数去做目标检测，简化了以往目标检测非极大值抑制的部分

训练：
CNN抽特征 -> Transformer Encoder 全局特征 -> Decoder 得到固定数量预测框 -> 与Ground truth 做二分图匹配（100个框 2个框匹配上了 剩下的标记为背景）算loss

预测：
最后一步卡置信度

结论：
COCO数据集上取得和Faster RCNN 差不多的效果，小物体上表现不怎么好，DETR训练太慢，能够作为统一的框架拓展到别的任务上（目标追踪，语义分割，视频里的姿态预测···）
