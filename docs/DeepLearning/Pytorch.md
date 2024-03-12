# PyTorch

## PyTorch基础

deep-learning-computation章节

## Tensor

### broadcast

broadcast的条件：

1. 每个Tensor至少有一个维度
2. 迭代维度尺寸时，从尾部开始，依次每个维度尺寸必须满足一下条件之一：
   - 相等
   - 其中一个tensor的维度为1
   - 其中一个tensor的维度不存在

### PyTorch自动求导

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
- **contiguous** 参考[博客](https://blog.csdn.net/kdongyi/article/details/108180250)

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

## Multiple GPUs

### 拆分数据

这种方式下，所有GPU尽管有不同的观测结果，但是执行着相同类型的工作。在完成每个小批量数据的训练之后，梯度在GPU上聚合。
GPU的数量越多，小批量包含的数据量就越大，从而就能提高训练效率。
缺点：不能够训练更大的模型

$k$个GPU并行训练过程如下：
*在任何一次训练迭代中，给定的随机的小批量样本都将被分成$k$个部分，并均匀地分配到GPU上；
*每个GPU根据分配给它的小批量子集，计算模型参数的损失和梯度；
*将$k$个GPU中的局部梯度聚合，以获得当前小批量的随机梯度；
*聚合梯度被重新分发到每个GPU中；
*每个GPU使用这个小批量随机梯度，来更新它所维护的完整的模型参数集。

向多个设备复制参数

```python

def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

能够将所有设备上的梯度进行相加

```python

def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

```

分发数据 `nn.parallel.scatter(data, devices)`

```python
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

```text
input : tensor([ [ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19] ])
load into [device(type='cuda', index=0), device(type='cuda', index=1)]
output: (tensor([ [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9] ], device='cuda:0'), tensor([ [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19] ], device='cuda:1'))
```

分发数据和标签

```python

def split_batch(X, y, devices):
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

实现多GPU训练

```python

def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # 在每个GPU上分别计算损失
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # 反向传播在每个GPU上分别执行
        l.backward()
    # 将每个GPU的所有梯度相加，并将其广播到所有GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # 在每个GPU上分别更新模型参数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 在这里，我们使用全尺寸的小批量
```

评估模型的时候只在一个GPU上

```python

def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到num_gpus个GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 为单个小批量执行多GPU训练
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # 在GPU0上评估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')
```

简洁实现`net = nn.DataParallel(net, device_ids=devices)`，和之前几乎没什么区别

```python

def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # 在多个GPU上设置模型
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')
```
