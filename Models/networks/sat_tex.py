import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
from Models.modules.multihead_isa_pool_attention import InterlacedPoolAttention
from functools import partial

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1,
                 has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        if mode == '2d':
            self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm2d
        elif mode == '1d':
            self.conv = nn.Conv1d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm1d
        if self.has_bn:
            self.bn = norm_layer(c_out)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

class KSCO_1(nn.Module):
    def __init__(self, level_num):
        super(KSCO_1, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBNReLU(256, 128, 3, 1, 1, has_relu=False),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = ConvBNReLU(128, 1, 1, 1, 0, has_bn=False, has_relu=False)
        self.f1 = nn.Sequential(
            ConvBNReLU(2, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'),
            nn.LeakyReLU(inplace=True)
        )
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')
        self.out = ConvBNReLU(128, 256, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num

        self.gate = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.attn = InterlacedPoolAttention(
            128,
            num_heads=2,
            window_size=7,
            rpe=True,
            dropout=0.0,
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(128)

        # ⚡ sửa: dùng 1D pool thay cho 2D
        self.avg_pool = nn.AdaptiveAvgPool1d(256)

    def forward(self, x):
        x1 = self.conv1(x)
        B, C, H, W  = x1.shape

        # attention branch
        x2 = x1.view(B, C, -1).permute(0, 2, 1)        # (B, N, C)
        str = self.attn(self.norm1(x2), H, W)          # (B, N, C)
        str = str.permute(0, 2, 1)                     # (B, C, N)
        str = self.avg_pool(str)                       # (B, C, 256)
        str = str.mean(dim=1, keepdim=True)            # (B, 1, 256)

        # conv2 branch
        x = self.conv2(x1)
        N, C, H, W = x.shape
        x = torch.sigmoid(x)
        cos_sim = x.view(N, -1)                        # (B, 256)

        # tính K
        K = []
        for i in cos_sim:
            s = pd.Series(i.cpu().detach().numpy())
            kurt = s.kurt()
            kurt = np.max(kurt, 0)
            K.append(kurt)
        K = np.vstack(K)
        K = torch.from_numpy(K).to(x.device)
        K = K.unsqueeze(-1)                            # (B, 1, 1)

        # quantization
        cos_sim_min, _ = cos_sim.min(-1)
        cos_sim_min = cos_sim_min.unsqueeze(-1)
        cos_sim_max, _ = cos_sim.max(-1)
        cos_sim_max = cos_sim_max.unsqueeze(-1)

        q_levels = torch.arange(self.level_num, device=x.device, dtype=torch.float32)
        q_levels = q_levels.expand(N, self.level_num)
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * \
                   (cos_sim_max - cos_sim_min) + cos_sim_min
        q_levels = q_levels.unsqueeze(1)
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]
        q_levels_inter = q_levels_inter.unsqueeze(-1)
        cos_sim = cos_sim.unsqueeze(-1)

        quant = (1 - torch.abs(q_levels - cos_sim))
        quant = abs(K) / torch.exp(quant)
        quant = quant * (quant > (1 - q_levels_inter))

        sta = quant.sum(1).unsqueeze(1)                # (B, 1, 256)

        # ⚡ giờ str và sta đều (B, 1, 256)
        sta = torch.cat([self.gate * str, sta], dim=1) # (B, 2, 256)

        sta = self.f1(sta)
        sta = self.f2(sta)
        sta = self.out(sta)
        return sta, quant

class KSCO_2(nn.Module):
    def __init__(self, level_num,in_dim):
        super(KSCO_2, self).__init__()
        self.conv1 = nn.Sequential(ConvBNReLU(in_dim, 128, 3, 1, 1, has_relu=False), nn.LeakyReLU(inplace=True))
        self.conv2 = ConvBNReLU(128, 1, 1, 1, 0, has_bn=False, has_relu=False)
        self.f1 = nn.Sequential(ConvBNReLU(1, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'),
                                nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')
        self.out = ConvBNReLU(128, 256, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num
    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x1)
        x = F.sigmoid(x)
        N, C, H, W = x.shape
        cos_sim = x.view(N, -1)
        K= []
        for i in cos_sim:
            s = pd.Series(i.cpu().detach().numpy())
            kurt = s.kurt()
            kurt = np.max(kurt,0)
            K.append(kurt)
        K = np.vstack(K)
        K = torch.from_numpy(K)
        #softmax
        #print(K.shape)
        #torch.Size([16, 1])

        cos_sim_min, _ = cos_sim.min(-1)
        cos_sim_min = cos_sim_min.unsqueeze(-1)
        cos_sim_max, _ = cos_sim.max(-1)
        cos_sim_max = cos_sim_max.unsqueeze(-1)
        #print(cos_sim_max.shape)torch.Size([16, 1])
        q_levels = torch.arange(self.level_num, device=x.device, dtype=torch.float32)
        q_levels = q_levels.expand(N, self.level_num)
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min
        q_levels = q_levels.unsqueeze(1)
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]

        q_levels_inter = q_levels_inter.unsqueeze(-1)
        cos_sim = cos_sim.unsqueeze(-1)

        K = K.unsqueeze(-1).to(x.device)
        #print(K.shape) torch.Size([16, 1, 1])
        quant = (1 - torch.abs(q_levels - cos_sim))#torch.max(K,0)
        #print(quant.shape)torch.Size([16, 1024, 64])
        quant =abs(K) /torch.exp(quant)
        quant = quant * (quant > (1 - q_levels_inter))
        #print(quant.shape)#torch.Size([16, 1024, 64])
        #quant = quant.sum(1)
        #quant = quant.unsqueeze(1)
        #print(quant.shape)torch.Size([16, 1, 64])

        return quant
