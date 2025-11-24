#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

# DCNv3 from OpenGVLab
from dcnv3 import dcnv3_core_pytorch as dcnv3


class ModulatedDeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride, padding, dilation, groups, deformable_groups, im2col_step):

        # Standardize params
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        kernel_size = _pair(weight.shape[2:4])

        output = dcnv3(
            input,
            offset,
            mask,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            deformable_groups
        )

        # Save for backward
        ctx.save_for_backward(input, offset, mask, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.kernel_size = kernel_size
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        DCNv3 has a built-in autograd implementation.
        Just call autograd.grad.
        """
        input, offset, mask, weight, bias = ctx.saved_tensors

        # Let PyTorch autograd compute gradients
        grads = torch.autograd.grad(
            outputs=dcnv3(
                input,
                offset,
                mask,
                weight,
                bias,
                ctx.stride,
                ctx.padding,
                ctx.dilation,
                ctx.groups,
                ctx.deformable_groups
            ),
            inputs=(input, offset, mask, weight, bias),
            grad_outputs=grad_output,
            allow_unused=True,
            retain_graph=False
        )

        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = grads

        return (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
            None, None, None, None, None, None
        )
