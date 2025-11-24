import torch
from torch import nn
from .dcn import DeformableConv2d

class ModulatedDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(ModulatedDeformConv, self).__init__()
        
        self.dcn = DeformableConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, input, offset, mask):
        return self.dcn(input, offset, mask)