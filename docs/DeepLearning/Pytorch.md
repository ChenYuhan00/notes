# Pytorch

## Pytorch基础

deep-learning-computation章节

### Pytorch自动求导

[深入浅出Pytroch 第二章](https://datawhalechina.github.io/thorough-pytorch/第二章/2.2%20自动求导.html)

#### 计算图

将代码分解成操作子，将计算表示为一个无环图
一般只有leaf node有grad

最后的输出看成关于网络权重的函数，backward函数计算出权重的梯度（全微分）
>对于leaf和require_grad的节点不能够进行inplace operation

>**retain_graph** 防止backward之后释放相关内存

**.detach** return a new tensor ,detached from the current graph,the result will never require gradient
将计算移动到计算图之外，当作常数

### 数据读取

#### 小批量随机梯度下降

随机采样b个样本 $i_1,i_2,...,i_b$来近似损失
$$
    \frac{1}{b}\sum_{i \in I_b}l(\hat y_i,y_i)
$$
一次迭代用b个数据计算后更新参数
一个epoch将数据集的数据都用一遍

#### data iterator

一次随机读取batch_size个数据

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

### 补充

- **cat**和**stack** 参考[博客](https://blog.csdn.net/twelve13/article/details/109728210)
- 批量矩阵乘法 torch.bmm(X,Y),X的shape为(n,a,b)，Y的shape为(n,b,c)，输出形状(n,a,c)
- **torch.unsqueeze()** 在指定纬度插入纬度1
  - X.shape = (2,3) X.unsqueeze(0).shape = (1，2，3) X.unsqueeze(1).shape = (2,1,3)
- **view 和 reshape** 参考[博客](https://zhuanlan.zhihu.com/p/436892343)

## Tools

### torchsummay

可以用来打印网络的结构参数

用法示例

```python
from torchsummary import summary
from torchvision.models import resnet18

model = resnet18()
summary(model, input_size=[(3, 256, 256)], batch_size=2, device="cpu")
```

```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [2, 64, 128, 128]           9,408
       BatchNorm2d-2          [2, 64, 128, 128]             128
              ReLU-3          [2, 64, 128, 128]               0
         MaxPool2d-4            [2, 64, 64, 64]               0
            Conv2d-5            [2, 64, 64, 64]          36,864
       BatchNorm2d-6            [2, 64, 64, 64]             128
              ReLU-7            [2, 64, 64, 64]               0
            Conv2d-8            [2, 64, 64, 64]          36,864
       BatchNorm2d-9            [2, 64, 64, 64]             128
             ReLU-10            [2, 64, 64, 64]               0
       BasicBlock-11            [2, 64, 64, 64]               0
           Conv2d-12            [2, 64, 64, 64]          36,864
      BatchNorm2d-13            [2, 64, 64, 64]             128
             ReLU-14            [2, 64, 64, 64]               0
           Conv2d-15            [2, 64, 64, 64]          36,864
      BatchNorm2d-16            [2, 64, 64, 64]             128
             ReLU-17            [2, 64, 64, 64]               0
       BasicBlock-18            [2, 64, 64, 64]               0
           Conv2d-19           [2, 128, 32, 32]          73,728
      BatchNorm2d-20           [2, 128, 32, 32]             256
             ReLU-21           [2, 128, 32, 32]               0
           Conv2d-22           [2, 128, 32, 32]         147,456
...
Forward/backward pass size (MB): 164.02
Params size (MB): 44.59
Estimated Total Size (MB): 210.12
----------------------------------------------------------------
```
