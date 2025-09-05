import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Conv Block =====
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, drop_out=False):
        super(conv_block, self).__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if drop_out:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ===== Up + Concat (chuẩn UNet) =====
class UpCat(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=True):
        super(UpCat, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, high_feat, low_feat):
        up_feat = self.up(high_feat)
        # pad cho khớp size
        diffY = low_feat.size()[2] - up_feat.size()[2]
        diffX = low_feat.size()[3] - up_feat.size()[3]
        up_feat = F.pad(up_feat, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([low_feat, up_feat], dim=1)
        return self.conv(x)


# ===== Up + Conv (không concat) =====
class UpCatconv(nn.Module): 
    def __init__(self, in_ch, out_ch, is_deconv=True):
        super(UpCatconv, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = conv_block(out_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


# ===== Biến thể UpCat2 (thêm conv sau concat) =====
class UpCat2(nn.Module):
    def __init__(self, in_ch, out_ch, is_deconv=True):
        super(UpCat2, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = conv_block(in_ch, out_ch)
        self.conv2 = conv_block(out_ch, out_ch)

    def forward(self, high_feat, low_feat):
        up_feat = self.up(high_feat)
        diffY = low_feat.size()[2] - up_feat.size()[2]
        diffX = low_feat.size()[3] - up_feat.size()[3]
        up_feat = F.pad(up_feat, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([low_feat, up_feat], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


# ===== Deep Supervision =====
class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=1)
        self.up = nn.Upsample(size=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x


# ===== Gating Signal (cho Attention UNet) =====
class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=1, is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()
        if is_batchnorm:
            self.gate = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.gate = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.gate(x)
