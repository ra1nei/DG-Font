import torch
from torch import nn
import torchvision.ops
import math

class ModulatedDeformConvFunction(object):
    
    @staticmethod
    def apply(input, offset, mask, weight, bias, 
              stride, padding, dilation, groups, deformable_groups, im2col_step):
        
        # im2col_step là tham số tối ưu bộ nhớ của repo cũ, 
        # torchvision tự động xử lý nên ta không cần dùng nhưng vẫn giữ tham số để không lỗi code gọi.
        
        # Chuyển đổi stride, padding, dilation về dạng tuple nếu cần (giống _pair)
        # torchvision nhận cả int và tuple nên thường không cần convert gắt gao,
        # nhưng để an toàn ta cứ truyền thẳng vào.
        
        return torchvision.ops.deform_conv2d(
            input=input,
            offset=offset,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            mask=mask
        )

class ModulatedDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        
        # Weight shape: [out_channels, in_channels // groups, *kernel_size]
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, offset, mask):
        return torchvision.ops.deform_conv2d(
            input, 
            offset, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            mask=mask
        )

    @property
    def in_channels(self):
        return self.weight.shape[1] * self.groups