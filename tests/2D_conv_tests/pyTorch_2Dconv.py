import numpy as np
import torch

import joey as ml
from joey.net import Net


def pyTorch_conv(input, kernel, stride, padding):

    weights = torch.randn(1,1, len(kernel), len(kernel))
    weights[0][0] = torch.Tensor(kernel)

    custom_input_T = torch.randn(1, 1, len(input), len(input))


    custom_input_T[0][0]= torch.Tensor(input);


    torch_conv_op = torch.nn.Conv2d(1,1, kernel_size=len(kernel), padding=padding , stride=stride)



    with torch.no_grad():
        
        torch_conv_op.weight = torch.nn.Parameter(weights)
        torch_conv_op.bias = torch.nn.Parameter(torch.Tensor([0]))

    result = torch_conv_op(custom_input_T).detach().numpy();

    return result
   