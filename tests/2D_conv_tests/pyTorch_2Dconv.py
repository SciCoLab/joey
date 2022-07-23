import numpy as np
import torch

import joey as ml
from joey.net import Net


def pyTorch_conv(input, kernel, padding, stride):

    weights = torch.randn(1,len(kernel), len(kernel[0]), len(kernel[0]))
    weights[0] = torch.Tensor(kernel)

    custom_input_T = torch.randn(1, len(input), len(input[0]), len(input[0]))

    custom_input_T[0]= torch.Tensor(input);

    torch_conv_op = torch.nn.Conv2d(1,len(kernel), kernel_size=len(kernel), padding=padding , stride=stride)

    with torch.no_grad():
        
        torch_conv_op.weight = torch.nn.Parameter(weights)
        torch_conv_op.bias = torch.nn.Parameter(torch.Tensor([0]))

    result = torch_conv_op(custom_input_T).detach().numpy();

    return result
   