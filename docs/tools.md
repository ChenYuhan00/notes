# Tools

## torch

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
