from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# 输入: [Batch, inplanes, H, W]
# |
# ├── 1×1 Conv（降维）→ BN → ReLU
# |
# ├── 3×3 Conv（提取特征）→ BN → ReLU
# |
# ├── 1×1 Conv（升维）→ BN
# |
# ├── AvgPool (如果 stride > 1)
# |
# ├── 跳跃连接（downsample）
# |
# ├── Add（残差连接）→ ReLU
# |
# 输出: [Batch, 4 × planes, H/stride, W/stride]

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


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype    # 记录输入数据的原始数据类型
        ret = super().forward(x.type(torch.float32))    # 将输入转换为 float32 计算 LayerNorm
        return ret.type(orig_type)  # 计算完后转换回原始数据类型

#（快速 GELU 激活函数）
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# 输入 x
#    │
#    ├── LayerNorm (ln_1)
#    │
#    ├── Multi-Head Attention（自注意力）
#    │
#    ├── 残差连接（x = x + attention_output）
#    │
#    ├── LayerNorm (ln_2)
#    │
#    ├── MLP（前馈神经网络）
#    │
#    ├── 残差连接（x = x + mlp_output）
#    │
#    └── 输出
class ResidualAttentionBlock(nn.Module):


    #d_model：输入的特征维度（隐藏维度）。
    #n_head：多头注意力的头数。
    #attn_mask：可选的 注意力掩码（Attention Mask），用于 屏蔽部分注意力计算（如自回归任务）。
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)  #是 PyTorch 内置的多头注意力实现
        self.ln_1 = LayerNorm(d_model)                      #归一化 d_model 维度上的数据，使训练更稳定，防止梯度消失/爆炸

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),  #把 d_model 扩展到 4×d_model 大小（扩展特征维度）
            ("gelu", QuickGELU()),                      #激活函数 QuickGELU()
            ("c_proj", nn.Linear(d_model * 4, d_model)) #把 4×d_model 维度投影回 d_model，确保残差连接维度匹配
        ]))

        self.ln_2 = LayerNorm(d_model)  #在 MLP 计算前，对输入进行层归一化，稳定训练。
        self.attn_mask = attn_mask      #注意力掩码

    def attention(self, x: torch.Tensor):

        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None  #确保 attn_mask 和 x 的数据类型和设备匹配。
        #query = x, key = x, value = x（典型的 自注意力计算）
        #need_weights=False：不返回注意力权重（只返回 output）
        #attn_mask：如果 attn_mask 存在，就应用掩码。
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



# 输入 x（序列数据）
#    │
#    ├── ResidualAttentionBlock #1（自注意力 + MLP + 残差）
#    │
#    ├── ResidualAttentionBlock #2（自注意力 + MLP + 残差）
#    │
#    ├── ...
#    │
#    ├── ResidualAttentionBlock #N（自注意力 + MLP + 残差）
#    │
#    └── 输出

# width：隐藏层的特征维度（等同于 d_model）。
# layers：Transformer Block 的数量（即堆叠多少层 ResidualAttentionBlock）。
# heads：多头注意力的头数。
# attn_mask：可选的 注意力掩码（用于屏蔽部分注意力计算）。
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        # 把 layers 个 ResidualAttentionBlock（残差注意力块） 叠加成一个完整的 Transformer 结构。
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



# input_resolution: 输入图像的分辨率（假设为正方形，例如 224x224）。
# patch_size: 图像被分割成的小块的大小（例如 16x16）。
# width: Transformer 的隐藏层维度（即每个 patch 的嵌入维度）。
# layers: Transformer 的层数。
# heads: Transformer 的多头注意力机制的头数。
# output_dim: 输出的特征维度。

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        #这是一个卷积层，用于将输入图像分割成 patches，并将每个 patch 转换为一个向量。
        #输入：[batch_size, 3, input_resolution, input_resolution] -> 输出:[batch_size, width, grid, grid] 其中grid = input_resolution // patch_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        #这是一个可学习的参数，用于表示分类任务的类别信息，它会被添加到 patch 嵌入序列的开头。
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        #这是一个可学习的位置嵌入，用于为每个 patch 添加位置信息。形状：[num_patches + 1, width]
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))

        #用于在输入 Transformer 之前对 patch 嵌入进行归一化
        self.ln_pre = LayerNorm(width)
        #输入形状为 [sequence_length, batch_size, width]，输出形状相同。
        self.transformer = Transformer(width, layers, heads)
        #用于在 Transformer 输出之后对特征进行归一化。
        self.ln_post = LayerNorm(width)
        #self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
