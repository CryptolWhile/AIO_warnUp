import torch
import torch.nn as nn
import torch.nn.functional as F

class scale_atten_convblock(nn.Module):
    """
    Scale Attention Layer: học trọng số trên nhiều feature maps khác nhau.
    """
    def __init__(self, in_size, out_size):
        super(scale_atten_convblock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_size)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        attn = self.sigmoid(self.conv2(x))
        return x * attn
