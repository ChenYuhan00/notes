---
title: Deep Learning_2
---

### Vision Transformer

[论文](https://arxiv.org/abs/2010.11929)
[代码来源](https://zhuanlan.zhihu.com/p/418184940)

主要就是图片转换tokens，position embedding这里是可学习的参数

>Example
输入224x224x3 -> Embedding(16x16的卷积核，步距为16的卷积层) 14x14x768 -> Flatten 196x768 -> Concat一个Class token 197x768 -> 加上position embedding -> Dropout -> 重复L次 Transformer Encoder 197x768-> LayerNorm -> Extract class token 1x768(之前concat的class token)-> MLP Head

```python

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size  # 每个patch的大小
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 224/16 -> 14*14
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # patches的数目
 
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)  # 卷积核大小和patch_size都是16*16
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()  # 如果没有传入norm层，就使用identity
 
    def forward(self, x):
        B, C, H, W = x.shape  # 注意，在vit模型中输入大小必须是固定的，高宽和设定值相同
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
 
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
```

```python

class Attention(nn.Module):  # Multi-head selfAttention 模块
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,  # head的个数
                 qkv_bias=False,  # 生成qkv时是否使用偏置
                 qk_scale=None,
                 attn_drop_ratio=0.,  # 两个dropout ratio
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个head的dim
        self.scale = qk_scale or head_dim ** -0.5  # 不去传入qkscale，也就是1/√dim_k
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 使用一个全连接层，一次得到qkv
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # 把多个head进行Concat操作，然后通过Wo映射，这里用全连接层代替
        self.proj_drop = nn.Dropout(proj_drop_ratio)
 
    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim] 加1代表类别，针对ViT-B/16，dim是768
        B, N, C = x.shape
 
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3（代表qkv）, num_heads（代表head数）, embed_dim_per_head（每个head的qkv维度）]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
 
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 每个header的q和k相乘，除以√dim_k（相当于norm处理）
        attn = attn.softmax(dim=-1)  # 通过softmax处理（相当于对每一行的数据softmax）
        attn = self.attn_drop(attn)  # dropOut层
 
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 得到的结果和V矩阵相乘（加权求和），reshape相当于把head拼接
        x = self.proj(x)  # 通过全连接进行映射（相当于乘论文中的Wo）
        x = self.proj_drop(x)  # dropOut
        return x
```

```python
class Mlp(nn.Module):  # Encoder中的MLP Block
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 如果没有传入out features，就默认是in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()  # 默认是GELU激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

```python
# Transformer Encoder层代码解读
class Block(nn.Module):  # Encoder Block
    def __init__(self,
                 dim,  # 每个token的维度
                 num_heads,  # head个数
                 mlp_ratio=4.,  # 第一个结点个数是输入节点的四倍
                 qkv_bias=False,  # 是否使用bias
                 qk_scale=None,
                 drop_ratio=0.,  # Attention模块中最后一个全连接层使用的drop_ratio
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)  # layer norm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)  # Multihead Attention
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # MLP第一个全连接层的个数
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
 
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
 
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1  # 一般等于1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # 为Norm传入默认参数
        act_layer = act_layer or nn.GELU
 
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)  # Patch Embedding层
        num_patches = self.patch_embed.num_patches  # patches的总个数
 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 构建可训练参数的0矩阵，用于类别
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  # 默认为None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # 位置embedding，和concat后的数据一样
        self.pos_drop = nn.Dropout(p=drop_ratio)  # DropOut层（添加了pos_embed之后）
 
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # 从0到ratio，有depth个元素的等差序列
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)  # 有多少层循环多少次
        ])
        self.norm = norm_layer(embed_dim)
 
        # Representation layer
        if representation_size and not distilled:  # representation_size为True就在MLPHead构建PreLogits,否则只有Linear层
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
 
        # Classifier head(s) 线性分类层
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
 
        # Weight init 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
 
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
 
    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768] 输入Patch Embedding层
        # [1, 1, 768] -> [B, 1, 768] 在batch维度复制batch_size份
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768] 将cls_token与x在维度1上拼接。注意：cls_token在前，x在后
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
 
        x = self.pos_drop(x + self.pos_embed)  # concat后的数据加上position，再经过dropout层
        x = self.blocks(x)  # 经过堆叠的Encoder blocks
        x = self.norm(x)  # Layer Norm层
        if self.dist_token is None:  # 一般为None，本质上是Identity层
            return self.pre_logits(x[:, 0])  # 提取cls_token信息，因为cls_token维度在前，所以索引为0就是cls本身
        else:
            return x[:, 0], x[:, 1]
 
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
```

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
