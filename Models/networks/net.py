import torch
import torch.nn as nn

from Models.layers.modules import conv_block, UpCat, UpCatconv, UnetDsv3, UnetGridGatingSignal3,UpCat2
from Models.layers.scale_attention_layer import scale_atten_convblock
from Models.modules.ftransformer import GeneralTransformerBlock
from Models.networks.sat_tex import KSCO_2,KSCO_1
import torch.nn.functional as F

# ===== Double Convolution Block =====
# Chuỗi Conv2d -> BN -> Softplus -> Conv2d -> BN -> Softplus
# Giúp tăng tính phi tuyến và trích xuất đặc trưng cục bộ
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Softplus(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Softplus()

        )

    def forward(self, x):
        x = self.conv(x)
        return x
        
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch , in_ch , 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1):
        x = x1
        x = self.conv(x)
        return x


class skinformer(nn.Module):
    def __init__(self, args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(skinformer, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = args.out_size

        # ================================
        # Feature Extraction (Encoder)
        # ================================
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Transformer block để fuse features (trên bottleneck)
        self.transformer = GeneralTransformerBlock(256, planes=256, num_heads=2)
        # downsampling
        # Encoder layers (Conv + MaxPool)
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Bottleneck
        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # attention blocks

        # ================================
        # Decoder (Upsampling + Skip Connections)
        # ================================
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)

        self.up4_layer = Up(filters[3], filters[3])
        self.up3_layer = Up(filters[2], filters[2])
        self.up2_layer = Up(filters[1], filters[1])
        self.up1_layer = Up(filters[0], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=2 * filters[0], out_channels=4, kernel_size=1)

         # Scale Attention để tái trọng số đặc trưng
        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        # final conv (without any concat)
        # Final output: segmentation map
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())
        
        # ================================
        # Quantization-based Co-occurrence (QCO) modules
        # ================================
        self.qco = KSCO_1(256)
        self.qco3 = KSCO_2(64, 64)
        self.qco2 = KSCO_2(64, 32)
        self.qco1 = KSCO_2(64, 16)

        # Các conv dùng để sinh heatmap đặc trưng
        self.heatmap = nn.Conv2d((30), 1, kernel_size=3, stride=1, padding=1)
        self.heatmap2 = nn.Conv2d((192), 256, kernel_size=3, stride=1, padding=1)
        self.heatmap3 = nn.Conv2d((256), 16, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down4 = nn.MaxPool2d(kernel_size=(4, 4))

    def forward(self, inputs):
        # ================================
        # Encoder
        # ================================
        conv1 = self.conv1(inputs)              
        # 256x256 -> torch.Size([16, 16, 256, 256])
        # 512x512 -> [B, 16, 512, 512]
        maxpool1 = self.maxpool1(conv1)         
        # 256x256 -> torch.Size([16, 16, 128, 128])
        # 512x512 -> [B, 16, 256, 256]

        conv2 = self.conv2(maxpool1)            
        # 256x256 -> torch.Size([16, 32, 128, 128])
        # 512x512 -> [B, 32, 256, 256]
        maxpool2 = self.maxpool2(conv2)         
        # 256x256 -> torch.Size([16, 32, 64, 64])
        # 512x512 -> [B, 32, 128, 128]

        conv3 = self.conv3(maxpool2)            
        # 256x256 -> torch.Size([16, 64, 64, 64])
        # 512x512 -> [B, 64, 128, 128]
        maxpool3 = self.maxpool3(conv3)         
        # 256x256 -> torch.Size([16, 64, 32, 32])
        # 512x512 -> [B, 64, 64, 64]

        conv4 = self.conv4(maxpool3)            
        # 256x256 -> torch.Size([16, 128, 32, 32])
        # 512x512 -> [B, 128, 64, 64]
        maxpool4 = self.maxpool4(conv4)         
        # 256x256 -> torch.Size([16, 128, 16, 16])
        # 512x512 -> [B, 128, 32, 32]

        center = self.center(maxpool4)          
        # 256x256 -> torch.Size([16, 256, 16, 16])
        # 512x512 -> [B, 256, 32, 32]

        # ================================
        # QCO modules
        # ================================
        sta, quant = self.qco(center)
        sta = sta.view(-1, 256, 16, 16) 
            
        # sta shape:  torch.Size([16, 256, 256])
        # quant shape:  torch.Size([16, 256, 256])
        # sta after shape:  torch.Size([16, 256, 16, 16])

        quant3 = self.qco3(maxpool3)            
        quant2 = self.qco2(maxpool2)            
        quant1 = self.qco1(maxpool1)            
        # quant3 shape:  torch.Size([16, 1024, 64])
        # quant2 shape:  torch.Size([16, 4096, 64])
        # quant1 shape:  torch.Size([16, 16384, 64])

        quant3 = quant3.view(-1, 64, 32, 32)
        quant2 = quant2.view(-1, 64, 64, 64)
        quant1 = quant1.view(-1, 64, 128, 128)
        # quant3 after shape:  torch.Size([16, 64, 32, 32])
        # quant2 after shape:  torch.Size([16, 64, 64, 64])
        # quant1 after shape:  torch.Size([16, 64, 128, 128])
        
        quant2 = self.down2(quant2)
        quant1 = self.down4(quant1)
        # quant2 shape:  torch.Size([16, 64, 32, 32])
        # quant1 shape:  torch.Size([16, 64, 32, 32])

        kv = self.heatmap2(torch.cat((quant3, quant2, quant1), dim=1))
        #kv shape:  torch.Size([16, 256, 32, 32])

        quant = quant.view(-1, 256, 16, 16)
        # quant after shape:  torch.Size([16, 256, 16, 16])

        q = self.up(quant)                      
        #q shape:  torch.Size([16, 256, 32, 32])


        # ================================
        # Transformer Fusion
        # ================================
        y = self.transformer(q, kv)    
        #y shape:  torch.Size([16, 256, 32, 32])        


        # ================================
        # Decoder
        # ================================
        up4_input = self.up_concat4(conv4, sta)   # concat skip connection
        up4_output = self.up4_layer(up4_input)        # upsample + double conv
        
        up3_input = self.up_concat3(conv3, up4_output)   # concat skip connection
        up3_output = self.up3_layer(up3_input)        # upsample + double conv

        up2_input = self.up_concat2(conv2, up3_output)   # concat skip connection
        up2_output = self.up2_layer(up2_input)        # upsample + double conv


        up1_input = self.up_concat1(conv1, up2_output)   # concat skip connection
        up1_output = self.up1_layer(up1_input)        # upsample + double conv


        # ================================
        # Fusion với ST features
        # ================================
        ST = self.heatmap3(y)                   
        # ST shape:  torch.Size([16, 16, 32, 32])
        ST = F.interpolate(ST, size=(256, 256), mode='bilinear', align_corners=True)
        # ST after shape:  torch.Size([16, 16, 256, 256])
        up1 = torch.cat((ST, up1_output), dim=1)       
        # up1 shape:  torch.Size([16, 32, 256, 256])

        # ================================
        # Deep Supervision + Output
        # ================================
        dsv1 = self.dsv1(up1)                   
        # 256x256 -> torch.Size([16, 4, 256, 256])
        out = self.final(dsv1)                  
        # 256x256 -> torch.Size([16, 2, 256, 256])

        return out

