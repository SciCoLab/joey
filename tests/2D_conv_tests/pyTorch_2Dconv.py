import numpy as np
import torch

import joey as ml
from joey.net import Net

'''
    Input should be a 3D matrix like 3x512x512 image where 3 is the channels and 512x512 is the size of image.
    Similary kernel should be be 3D matrix where 3x5x5, 3 is the channel which 
    should always match the channels in input image, 5x5 is the kernel filter.
    Assumptioms - kernel would be of square size like 5x5, 3x3, same goes for image
'''

def pyTorch_conv(input, kernel, padding, stride):

    weights = torch.randn(len(kernel),len(kernel[0]), len(kernel[0][0]), len(kernel[0][0]))
    weights = torch.Tensor(kernel)

    custom_input_T = torch.randn(len(input), len(input[0]), len(input[0][0]), len(input[0][0]))

    custom_input_T= torch.Tensor(input);

    torch_conv_op = torch.nn.Conv2d(len(input[0]),len(kernel), kernel_size=len(kernel[0][0]), padding=padding , stride=stride)

    with torch.no_grad():
        
        torch_conv_op.weight = torch.nn.Parameter(weights)
        torch_conv_op.bias = torch.nn.Parameter(torch.Tensor([0]*len(kernel)))

    result = torch_conv_op(custom_input_T).detach().numpy();

    return result
   