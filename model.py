from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



class Bottleneck(nn.Module):
    expansion = 4 #扩展因子，表示通道数在 conv3 之后会扩展 4 倍



    #inplanes 输入通道数
    #planes 中间通道数（Conv2 和 Conv3 之前的通道数）
    #stride=1：默认步长为 1，如果 stride > 1，则表示需要进行 下采样（降低分辨率）
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()


        #卷积层
        #第一层conv1：1x1（减少通道数，降低计算量） 归一化 ReLU激活
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        #第二层conv2：3x3（提取特征）归一化 ReLU激活
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        #如果 stride > 1，用于降采样,平均池化
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        #第三层conv2：1x1（增加通道数 planes × 4） 归一化 ReLU激活
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)


        #下采样层
        self.downsample = None
        self.stride = stride

        #当输入x的通道数inplanes≠输出通道数planes × 4，表面残差链接时x的维度和out不一致，需要将两者下采样调整一致，才能相加。
        # 或者 步长 stride > 1 时，需要跳跃连接（shortcut）进行下采样匹配
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))


    #前向传播
    def forward(self, x: torch.Tensor):
        #保存原始输入x，用于残差连接
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        #如果 downsample 存在，使用 downsample 处理 identity 以匹配 out 的维度
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity #残差链接
        out = self.relu3(out)   #ReLU 激活（增强非线性表达能力）
        return out




#AttentionPool2d是一个 自注意力池化层（Attention Pooling Layer），用于将二维特征图 转换为全局表示（Global Representation）。
class AttentionPool2d(nn.Module):

    #spacial_dim：输入特征图的空间尺寸（假设 spacial_dim=7，则输入特征图大小为 7×7）。
    #embed_dim：输入特征的通道数（Transformer 的嵌入维度）。
    #num_heads：多头注意力的头数（通常 num_heads = 8 或 num_heads = 12）。
    #output_dim：输出的特征维度（如果未指定，默认与 embed_dim 相同）。
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        #torch.randn(spacial_dim ** 2 + 1, embed_dim) 生成一个 随机初始化的可训练位置编码。
        # 维度：spacial_dim × spacial_dim -> spacial_dim*spacial_dim+1 × embed_dim
        #+1：额外添加一个 全局查询向量（类似于 CLS token），用于池化全局信息。
        #embed_dim ** 0.5 进行缩放，避免数值过大影响训练稳定性。
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)




        #q_proj（Query 投影）
        #k_proj（Key 投影）
        #v_proj（Value 投影）
        #这些 Linear 层用于 将输入投影到 Transformer 自注意力所需的 Q、K、V 形式。
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        #最终投影：将注意力机制的输出进行通道变换，确保输出特征维度匹配 output_dim（如果未指定，默认 output_dim=embed_dim）。
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads



    #输入 x：形状为 (N, C, H, W)，其中：
    #    N：批次大小（batch size）
    #    C：通道数（embed_dim）
    #    H, W：特征图的空间维度（spacial_dim × spacial_dim）
    def forward(self, x):

        x = (x.flatten(start_dim=2)     #将二维特征图(H, W)展平为一维，形状由(N, C, H, W)变为 (N, C, H×W)
             .permute(2, 0, 1))         #重新排列维度，将 x 变为(HW, N, C).


        #x.mean(dim=0, keepdim=True)：计算所有位置的平均值，得到一个全局特征向量（类似 CLS token）
        #torch.cat([...], dim=0)：将这个全局查询向量作为第一行拼接到 x，最终形状变为 (HW+1, N, C)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)

        #positional_embedding[:, None, :]：变换为 (HW+1, 1, C)，让每个批次都使用相同的位置编码。
        x = x + self.positional_embedding[:, None, :].to(x.dtype)


        x, _ = F.multi_head_attention_forward(
            query=x[:1], #只使用第一个 Global Query 作为 查询向量（类似 CLS token）
            key=x,       #整个 x 作为 Key
            value=x,     #整个 x 作为 Value
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        #x[:1] 形状为 (1, N, C)，去掉第一个维度，返回形状 (N, C)，即 批次大小 N 的全局特征。
        return x.squeeze(0)



class ModifiedResNet(nn.Module):

    # layers：定义 ResNet 的层数（例如 [3, 4, 6, 3] 对应 ResNet-50）。
    # output_dim：模型的最终输出维度。
    # heads：多头注意力的头数（用于 AttentionPool2d）。
    # input_resolution：输入图像尺寸，默认 224×224。  目前为200×200
    # width：控制网络的通道数，默认 64。
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        #平均池化AvgPool2d(2)降低分辨率，减少信息丢失
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width # 记录当前的输入通道数（可变变量）
        self.layer1 = self._make_layer(width, layers[0])    #通道数 64
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)  #通道数 128，下采样（分辨率减半）
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)  #通道数 256，下采样（分辨率减半）
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)  #通道数 512，下采样（分辨率减半）

        embed_dim = width * 32  # ResNet 的最终特征维度
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim) #AttentionPool2d 提取图像的全局特征


    #第一层 Bottleneck 负责 stride=2 的降采样（如果需要）
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 3 层 3×3 卷积，然后 AvgPool2d(2) 降采样。
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x


        x = x.type(self.conv1.weight.dtype) # 确保数据类型匹配
        x = stem(x)                         # 通过 stem 处理输入
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)                 # 使用 Attention Pooling 提取全局特征

        return x


