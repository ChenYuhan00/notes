# Deformable DETR

[Deformable DETR: Deformable Transformers For End-to-end Object Detection](https://arxiv.org/pdf/2010.04159.pdf)
（2020.8）

[官方代码](https://github.com/fundamentalvision/Deformable-DETR)

[参考博客](https://zhuanlan.zhihu.com/p/372116181)

DETR的问题：
1.收敛速度慢
2.小目标检测效果差

Deformable DETR结合和变形卷积和DETR解决DETR的问题

每个特征像素不必与所有的特征像素交互计算，只需与部分基于采样的其他像素交互即可，加快了模型的收敛

## DCN

可形变卷积公式

$$
    \mathbf{y}(\mathbf{p_0}) = \sum_{p_n \in \mathcal{R}} \mathbf{w}(\mathbf{p_n}) \mathbf{x}(\mathbf{p_0} + \mathbf{p_n} + \Delta\mathbf{p_n})
$$

[code]()

[来源](https://blog.csdn.net/justsolow/article/details/105971437)


## Deformable Transformers

### Multi-Head Attention

这里重写了一下多头注意力的计算公式

$$
\text{MultiHeadAttn}(z_q,\mathbf{x}) = \sum_{m=1}^{M} \mathbf{W}_m[\sum_{k \in \Omega_k} A_{mqk}\mathbf{W}_m^\prime x_k]
$$

和 $\text{softmax}(\frac{QK^T}{\sqrt{d}})V$那个的多头版的公式是等价的。[d2l-multihead](../DeepLearning.md###Mutihead-Attention)

其中 $q \in \Omega_q$ 表示query的index，$k \in \Omega_k$ 表示key和value的index（包括所有的HW个点），$z_q,x_k\in \mathbb{R}^C$，M表示头的个数，$\mathbf{W}_m^\prime \in \mathbb{R}^{C_v \times C}$，$\mathbf{W}_m \in \mathbb{R}^{C \times C_v}$ （$C_v =C / M$），attention的权重$A_{mqk} \propto \exp{(\frac{z_q^TU_m^TV_mx_k}{\sqrt{C_v}})}$，并且满足$\sum_{k \in \Omega_k}A_{mqk} = 1$ （softmax），$U_m,V_m \in \mathbb{R}^{C_v \times C}$。

这种计算方式有两个问题：一是收敛很慢，需要大量的训练，因为当 $N_k$ 很大的时候$A_{mqv}$接近于$1/N_k$导致输入特征的梯度模糊；二是注意力计算的复杂度很高，上式的计算复杂度为 $O(N_qC^2 + N_k C^2 + N_qN_kC)$（没搞懂怎么算的文章是这样），在图像领域一般有$N_q = N_k \gg C$，有复杂度为$O(N_qN_kC)$随着特征图大小的二次复杂度增长。

### Deformable Attention

给定一个 feature map $\mathbf{x} \in \mathbb{R}^{C \times H \times W}$,2-d的point $p_q$，这里$K$表示采样的点的个数，有 $HW \gg K$

$$
\text{DeformAttn}(z_q,\mathbf{x}) = \sum_{m=1}^{M} \mathbf{W}_m[\sum_{k=1}^K A_{mqk}\mathbf{W}_m^\prime \mathbf{x}(p_q + \Delta p_{mqk})]
$$

这里 $\Delta p_{mqk} \in \mathbb{R}^2$ 没有约束，由于$\mathbf{x}(p_q + \Delta p_{mqk})$ 并不一定是整数，使用了双线性插值计算。

### Multi-scale Deformable Attention

$$
\text{MSDeformAttn}(z_q,\hat{p}_q,\{\mathbf{x}^l\}^L_{l=1}) = \sum_{m=1}^{M} \mathbf{W}_m[\sum_{l=1}^L\sum_{k=1}^K A_{mlqk}\mathbf{W}_m^\prime \mathbf{x}^l(\phi_l(\hat{p_q}) + \Delta p_{mlqk})]
$$

L表示输入特征的层级，$\phi_l(\hat{p}_q)$ 将归一化坐标 $\hat{p}_q$ 重新缩放到第 $l$ 层级的特征图，$A_{mlqk}$ 满足 $\sum_{l=1}^L\sum_{k=1}^KA_{mlqk} = 1$

当 $L=K=1$，$W_m^\prime$ 为单位矩阵呢时相当于可变形卷积

### Encoder Decoder

用多尺度可形变注意力模块替换DETR中处理特征的Transformer Encoder

![Deformable-DETR_1](../img/DeepLearning/Deformable-DETR_1.png)

Decoder中只把cross-attention的模块替换为多尺度可变形注意力，self-attention保持不变

## Additional Inprovements

Iterative Bounding Box Refinement和Two-Stage Deformable DETR

（没看懂
