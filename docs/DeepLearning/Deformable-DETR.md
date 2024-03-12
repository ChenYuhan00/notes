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

[参考博客](https://blog.csdn.net/justsolow/article/details/105971437)

可形变卷积公式

$$
    \mathbf{y}(\mathbf{p_0}) = \sum_{p_n \in \mathcal{R}} \mathbf{w}(\mathbf{p_n}) \mathbf{x}(\mathbf{p_0} + \mathbf{p_n} + \Delta\mathbf{p_n})
$$

```python
#...
self.offsets = nn.Conv2d(128, 18, kernel_size=3, padding=1)
self.conv4 = DeformConv2D(128, 128, kernel_size=3, padding=1)
#...
offsets = self.offsets(x)
x = F.relu(self.conv4(x, offsets))
#...
```

对每个输入的特征图，比如使用3x3的卷积核，offset就是2x3x3=18（x，y），先过卷积层得到feature map上每个点对应的9个点的offset的xy，然后通过算出每个点对应的九个点的值（对9个点每个点用双线性插值算出值），然后把每个点对应的3x3的搞在一起，比如10x10的特征图变成30x30，然后过一个3x3的kernel，strike为3的卷积层。DCN可训练的参数就是得到offset的卷积核的参数以及最后那个卷积核的参数。这里对于所有进defromConv2D的特征图都用的同一个偏移量。

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

## Code

[注释版源代码](https://github.com/xunull/read-Deformable-DETR)

找到了一篇源码解析的[博客](https://www.jianshu.com/p/a1f4831a21b2)

```python
#class MSDeformAttn
def forward(self, query, reference_points,
                input_flatten, input_spatial_shapes,
                input_level_start_index,
                input_padding_mask=None):
        """
        query是上一层的输出加上了位置编码
        :param query                       (N, Length_{query}, C)
        参考点位的坐标
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        encoder是上一层的输出，decoder使用的是encoder的输出 [bs, all hw, 256]
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        4个特征层的高宽
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        各个特征层的起始index的下标 如: [    0,  8056, 10070, 10583]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        [bs,all hw]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        # query 是src+pos，query下面变成了attention_weights
        # input_flatten 是src，input_flatten 对应了V
        # bs, all hw（decoder 是300）, 256
        N, Len_q, _ = query.shape
        # [bs,all hw,256]
        N, Len_in, _ = input_flatten.shape

        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        # 对encoder上一层的输出，或者decoder使用的encoder的输出 进行一层全连接变换，channel不变
        value = self.value_proj(input_flatten)

        if input_padding_mask is not None:
            # 填充0 [bs, all hw,256]
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # 分成多头，拆分的是最后的256 [bs,all hw,256] -> [bs,all hw, 8, 32]
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # sampling_offsets 是一个全连接
        # like (bs, all hw,8,4,4,2) 8个头，4个特征层，4个采样点 2个偏移量坐标
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # attention_weights 是一个全连接
        # like (bs, all hw,8,16)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)

        # like (bs,all hw,8,4,4)
        # 经过softmax 保证权重和为1
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:

            # input_spatial_shapes 换位置，高宽 变成 宽高
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)

            # reference_points  [bs,all hw,4,2] -> [bs,all hw,1,4,1,2]
            # sampling_offsets  [bs,all hw,8,4,4,2]
            # offset_normalizer [4,2] -> [1,1,1,4,1,2]
            # like (bs, hw,8,4,4,2)
            # 采样点加上偏移量
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:

            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        # 这里调用了cuda实现
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index,
            sampling_locations, attention_weights,
            self.im2col_step)

        output = self.output_proj(output)
        return output
```

这个算到的是attention weight算出来了 $\mathbf{x}^l(\phi_l(\hat{p_q}) + \Delta p_{mlqk}$ 这个是通过`F.grid_sample()`双线性插值算出来的

```python
def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()
```
