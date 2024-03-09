# Computer Vision

## Image Augmentation

- 翻转（上下翻转、左右翻转）
`torchvision.transforms.RandomHorizontalFlip()`
`torchvision.transforms.RandomVerticalFlip()`
- 随机剪裁
`torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))`
- 改变颜色（亮度、对比度、饱和度、色调）
`torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)`

## 目标检测

### Bounding Box

两种表示

- 左上角坐标和右下角坐标
- 中心坐标和w，h

### 锚框

#### 生成锚框

输入图像的高度为$h$，宽度为$w$。以图像的每个像素为中心生成不同形状的锚框：*缩放比*为$s\in (0, 1]$，*宽高比*为$r > 0$。
那么**锚框的宽度和高度分别是$hs\sqrt{r}$和$hs/\sqrt{r}$。** 请注意，当中心位置给定时，已知宽和高的锚框是确定的。

缩放比（scale）取值$s_1,\ldots, s_n$和宽高比（aspect ratio）取值$r_1,\ldots, r_m$。使用这些比例和长宽比的所有组合以每个像素为中心时，输入图像将总共有$whnm$个锚框。
在实践中，考虑包含$s_1$或$r_1$的组合：

$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$

也就是说，以同一像素为中心的锚框的数量是$n+m-1$。对于整个输入图像，将共生成$wh(n+m-1)$个锚框。

#### 交并比(IoU)

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$

#### 锚框分配

每次取最大IoU的锚框和真实框，去掉之后重复，最后根据阈值确定是否为锚框分配真实框

#### 标记类别

给定框$A$和$B$，中心坐标分别为$(x_a, y_a)$和$(x_b, y_b)$，宽度分别为$w_a$和$w_b$，高度分别为$h_a$和$h_b$，可以将$A$的偏移量标记为：

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$

#### 非极大值抑制

在同一张图像中，所有预测的非背景边界框都按置信度降序排序，以生成列表$L$。

1. 从$L$中选取置信度最高的预测边界框$B_1$作为基准，然后将所有与$B_1$的IoU超过预定阈值$\epsilon$的非基准预测边界框从$L$中移除。
2. 从$L$中选取置信度第二高的预测边界框$B_2$作为又一个基准，然后将所有与$B_2$的IoU大于$\epsilon$的非基准预测边界框从$L$中移除。
3. 重复上述过程，直到$L$中的所有预测边界框都曾被用作基准。此时，$L$中任意一对预测边界框的IoU都小于阈值$\epsilon$；因此，没有一对边界框过于相似。
4. 输出列表$L$中的所有预测边界框。

### R-CNN

#### Region-based CNN
1. 使用启发式搜索选择锚框
2. 使用预训练模型对每个锚框抽取特征
3. 训练一个SVM来对类别进行分类
4. 训练一个线性回归模型来预测边缘框偏移

**Rol Pooling**
给定一个锚框，均匀分成$n \times m$块，输出每块里面的最大值，不管锚框大小为多少，都是nm

#### Fast RCNN

使用CNN对图片抽取特征（整张图片），使用Rol Pooling层对每个锚框生成固定长度的特征

#### Faster RCNN

使用一个网络来替代启发式搜索来获得更好的锚框

#### Mask RCNN

如果有像素级别的标号，用FCN来利用这些信息

### YOLO

只看一次
- SSD锚框大量重叠浪费计算
- YOLO将图片均匀分成$S \times S$ 锚框
- 每个锚框预测$B$个边缘框

### 单发多框检测（SSD）

## 语义分割

图像分割和实例分割

### 转置卷积

转置卷积可以用来增加高宽

$Y[i: i + h, j: j + w] += X[i, j] * K$

**转置?**
$Y = X * W$
可以对W构造一个V，使得卷积等价于矩阵乘法$Y^ \prime = V X^ \prime$, 这里$X^ \prime Y^\prime$是 $XY$对应的向量版本
转置卷积等价于 $Y^\prime = V^T X^\prime $

基本操作

```python

def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

```

torch API

```python

X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
```

```text
tensor([[[[ 0.,  0.,  1.],
          [ 0.,  4.,  6.],
          [ 4., 12.,  9.]]]])
```

`tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)`padding将删除第一和最后一行和列

```text
tensor([[[[4.]]]])
```

`tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)`
步幅为2增大输出

```text
tensor([[[[0., 0., 0., 1.],
          [0., 0., 2., 3.],
          [0., 2., 0., 3.],
          [4., 6., 6., 9.]]]])
```

### 全卷积网络(FCN)

用转置卷积层来替换CNN最后的全连接层，从而实现每个像素的预测

```text
CNN --> 1x1 Conv --> 转置卷积 --> output
```

先用Resnet18提取特征`net = nn.Sequential(*list(pretrained_net.children())[:-2])`

然后加1x1的卷积层和转置卷积层，使得输出大小和原图像大小相同

```python
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```
