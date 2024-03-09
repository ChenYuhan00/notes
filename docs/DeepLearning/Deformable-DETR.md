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

```python
class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    # 注意，offset的Tensor尺寸是[b, 18, h, w]，offset传入的其实就是每个像素点的坐标偏移，也就是一个坐标量，最终每个点的像素还需要这个坐标偏移和原图进行对应求出。
    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        # N=9=3x3
        N = offset.size(1) // 2
		
		#这里其实没必要，我们反正这个顺序是我们自己定义的，那我们直接按照[x1, x2, .... y1, y2, ...]定义不就好了。
        # 将offset的顺序从[x1, y1, x2, y2, ...] 改成[x1, x2, .... y1, y2, ...]
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        # torch.unsqueeze()是为了增加维度,使offsets_index维度等于offset
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        # 根据维度dim按照索引列表index将offset重新排序，得到[x1, x2, .... y1, y2, ...]这样顺序的offset
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        # 对输入x进行padding
        if self.padding:
            x = self.zero_padding(x)

        # 将offset放到网格上，也就是标定出每一个坐标位置
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # 维度变换
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # floor是向下取整
        q_lt = Variable(p.data, requires_grad=False).floor()
        # +1相当于向上取整，这里为什么不用向上取整函数呢？是因为如果正好是整数的话，向上取整跟向下取整就重合了，这是我们不想看到的。
        q_rb = q_lt + 1

        # 将lt限制在图像范围内，其中[..., :N]代表x坐标，[..., N:]代表y坐标
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # 将rb限制在图像范围内
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # 获得lb
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        # 获得rt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)
        
        # 限制在一定的区域内,其实这部分可以写的很简单。有点花里胡哨的感觉。。在numpy中这样写：
        #p = np.where(p >= 1, p, 0)
        #p = np.where(p <x.shape[2]-1, p, x.shape[2]-1)

        # 插值的时候需要考虑一下padding对原始索引的影响
        # (b, h, w, N)
        # torch.lt() 逐元素比较input和other，即是否input < other
        # torch.rt() 逐元素比较input和other，即是否input > other
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        #禁止反向传播
        mask = mask.detach()
		#p - (p - torch.floor(p))不就是torch.floor(p)呢。。。
        floor_p = p - (p - torch.floor(p))
        #总的来说就是把超出图像的偏移量向下取整
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)


        # bilinear kernel (b, h, w, N)
        # 插值的4个系数
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # 插值的最终操作在这里
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        #偏置点含有九个方向的偏置，_reshape_x_offset() 把每个点9个方向的偏置转化成 3×3 的形式，
        # 于是就可以用 3×3 stride=3 的卷积核进行 Deformable Convolution，
        # 它等价于使用 1×1 的正常卷积核（包含了这个点9个方向的 context）对原特征直接进行卷积。
        x_offset = self._reshape_x_offset(x_offset, ks)
        
        out = self.conv_kernel(x_offset)

        return out

    #求每个点的偏置方向
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    #求每个点的坐标
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    #求最后的偏置后的点=每个点的坐标+偏置方向+偏置
    def _get_p(self, offset, dtype):
        # N = 9, h, w
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    #求出p点周围四个点的像素
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)将图片压缩到1维，方便后面的按照index索引提取
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)这个目的就是将index索引均匀扩增到图片一样的h*w大小
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        
        #双线性插值法就是4个点再乘以对应与 p 点的距离。获得偏置点 p 的值，这个 p 点是 9 个方向的偏置所以最后的 x_offset 是 b×c×h×w×9。
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #_reshape_x_offset() 把每个点9个方向的偏置转化成 3×3 的形式
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
```

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
