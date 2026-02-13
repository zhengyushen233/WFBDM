import torch
from torch import nn
import numpy as np
from torchvision.transforms import ToTensor
from PIL import ImageFilter
from .unet_block.ConditionNet import ConditionNet
from basicsr.utils.registry import ARCH_REGISTRY
from torchvision.transforms import ToPILImage

import math


def np_to_pil(img_np):
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    return ToPILImage()(img_np)


def torch_to_np(img_var):
    return img_var.detach().cpu().numpy().transpose(1, 2, 0)


def get_A(x):
    processed_images = []
    device = x.device
    for i in range(x.size(0)):
        img_np = torch_to_np(x[i])
        img_pil = np_to_pil(img_np)
        h, w = img_pil.size
        windows = (h + w) / 2
        A_pil = img_pil.filter(ImageFilter.GaussianBlur(windows))
        A_tensor = ToTensor()(A_pil).to(device, non_blocking=True)
        processed_images.append(A_tensor)
    return torch.stack(processed_images, dim=0)


class TNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.aNet = ConditionNet(support_size=input_channels)
        self.final = nn.Conv2d(128, output_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        a = self.final(self.aNet(x))
        return a


@ARCH_REGISTRY.register()
class PPG(nn.Module):
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.tNet = TNet(input_channels, 1)
        self.aNet = TNet(input_channels, 3)

    def forward(self, x):
        a = self.aNet(get_A(x) + x)
        t = self.tNet(x)
        return a, t


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

        self.residual = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, input):
        residual = self.residual(input)
        out = self.depth_conv(input)
        out = self.point_conv(out)
        out += residual
        return out

class LightweightAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.fusion_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.high_low_weight = nn.Parameter(torch.ones(2))

    def forward(self, high_feat, low_feat):
        n, num_heads, h, w, head_dim = high_feat.shape
        high_feat = high_feat.permute(0, 1, 4, 2, 3).reshape(n, -1, h, w)
        low_feat = low_feat.permute(0, 1, 4, 2, 3).reshape(n, -1, h, w)
        high_feat = self.fusion_conv(high_feat)
        low_feat = self.fusion_conv(low_feat)

        weight = torch.softmax(self.high_low_weight, dim=0)
        fused_feat = weight[0] * high_feat + weight[1] * low_feat
        return high_feat, low_feat, fused_feat


@ARCH_REGISTRY.register()
class CFRM(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(CFRM, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.dim = dim

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.valueh = Depth_conv(in_ch=dim, out_ch=dim)
        self.valuel = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attn_fusion = LightweightAttentionFusion(dim, num_heads)

    def transpose_for_scores(self, x):
        n, c, h, w = x.shape
        x = x.view(n, self.num_heads, self.attention_head_size, h, w)
        return x.permute(0, 1, 3, 4, 2)

    def forward(self, hidden_states, ctx):
        n, c, h, w = hidden_states.shape
        ctx = ctx[:n] + ctx[n:2 * n] + ctx[2 * n:]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layerh = self.valueh(ctx)
        mixed_value_layerl = self.valuel(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layerh = self.transpose_for_scores(mixed_value_layerh)
        value_layerl = self.transpose_for_scores(mixed_value_layerl)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores /= math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        self.dropout(attention_probs)

        ctx_layerh = torch.matmul(attention_probs, value_layerh)
        ctx_layerl = torch.matmul(attention_probs, value_layerl)

        ctx_layerh, ctx_layerl, _ = self.attn_fusion(ctx_layerh, ctx_layerl)

        ctx_layerh = ctx_layerh.repeat(3, 1, 1, 1)
        ctx_layerl = ctx_layerl.contiguous()

        return ctx_layerh, ctx_layerl