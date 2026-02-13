import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.archs.arch_util import ResidualBlockNoBN, default_init_weights
from basicsr.archs.Padiff_arch.dft_arch import LayerNorm, FeedForward


class EnhancedSELayer(nn.Module):

    def __init__(self, channel, reduction=8, use_hsigmoid=True):
        super(EnhancedSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        )

        self.activation = nn.Hardsigmoid() if use_hsigmoid else nn.Sigmoid()

        self.residual = nn.Conv2d(channel, channel, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)
        y = avg_feat + max_feat

        y = self.fc(y)
        y = self.activation(y)

        return x * y + self.residual(x)

class INRMLP(nn.Module):
    """MLP for Implicit Neural Representation, used to model frequency features"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            layers.append(ResidualBlockNoBN(num_feat=hidden_dim))
        layers.append(nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1))
        self.mlp = nn.Sequential(*layers)
        default_init_weights(self.mlp, 0.1)

    def forward(self, x):
        return self.mlp(x)


class FrequencyMapping(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')
        self.attn = nn.Conv2d(dim, dim, kernel_size=1)
        self.ffn = FeedForward(dim, ffn_factor=2.0, bias=True)

    def forward(self, x):
        x = self.norm(x)
        attn_feat = self.attn(x)
        x = x + attn_feat
        x = x + self.ffn(x)
        return x


class INR(nn.Module):
    """Implicit Neural Representation module with enhanced attention"""

    def __init__(self, in_channel=3, out_channel=3, hidden_dim=64):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channel, hidden_dim, kernel_size=3, padding=1)

        self.freq_mapping1 = FrequencyMapping(hidden_dim)
        self.se1 = EnhancedSELayer(channel=hidden_dim)

        self.freq_mapping2 = FrequencyMapping(hidden_dim)
        self.se2 = EnhancedSELayer(channel=hidden_dim)

        self.inr_mlp = INRMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            num_layers=6
        )

        self.final_se = EnhancedSELayer(channel=hidden_dim)
        self.out_proj = nn.Conv2d(hidden_dim, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.in_proj(x)

        x = self.freq_mapping1(x)
        x = self.se1(x)

        x = self.freq_mapping2(x)
        x = self.se2(x)

        implicit_feat = self.inr_mlp(x)

        implicit_feat = self.final_se(implicit_feat)
        implicit_feat = self.out_proj(implicit_feat)

        return implicit_feat