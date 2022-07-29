from asyncio import FastChildWatcher
from asyncio.log import logger
from datetime import datetime
import pytest
import numpy as np
from sympy import true
import joey
import torch

from devito import configuration


configuration['language'] = 'openmp'
#configuration['LOGGING'] = 'debug'

torch.manual_seed(0)


def pyTorch_conv3D(input_data, kernel_data, padding, stride):

    input_size = input_data.size()
    kernel_size = kernel_data.size()
    torch_conv_op = torch.nn.Conv3d(input_size[1],kernel_size[0],
                    kernel_size=kernel_size[-1], padding=padding , stride=stride)

    with torch.no_grad():    
        torch_conv_op.weight = torch.nn.Parameter(kernel_data)
        torch_conv_op.bias = torch.nn.Parameter(torch.Tensor([0]*kernel_size[0]))

    result = torch_conv_op(input_data).detach().numpy()

    return result

def pyTorch_conv2D(input_data, kernel_data, padding, stride):

    input_size = input_data.size()
    kernel_size = kernel_data.size()
    torch_conv_op = torch.nn.Conv2d(input_size[1],kernel_size[0],
                    kernel_size=kernel_size[-1], padding=padding , stride=stride)

    torch_conv_op.weight = torch.nn.Parameter(kernel_data)
    torch_conv_op.bias = torch.nn.Parameter(torch.Tensor([0]*kernel_size[0]))

    result = torch_conv_op(input_data)

    return result.detach().numpy()


def generate_random_input(input_size, kernel_size) -> tuple:
    '''generate random data for test'''

    kernel = torch.randn(kernel_size[0],input_size[1]
                    ,kernel_size[-2],kernel_size[-1], dtype= torch.float64)
    input_data = \
        torch.randn(input_size[0],input_size[1],input_size[-2],input_size[-1], dtype= torch.float64)
    if len(input_size) == 5:
        kernel = torch.randn(kernel_size[0],input_size[1],kernel_size[-3],kernel_size[-2]
                            ,kernel_size[-1], dtype= torch.float64)
        input_data = \
            torch.randn(input_size[0],input_size[1],input_size[2],input_size[3],input_size[4], dtype= torch.float64)

    return input_data, kernel

@pytest.mark.parametrize("input_size, kernel_size, padding, stride",
[((2,3,5,5), (6,2,3), 5, 1),((1,3,48,50), (6,5,7), 10, 3) ])
def test_joey_pytorch_conv2d(input_size, kernel_size, padding, stride, print_results = False):
    ''' test function for 3d conv operation'''
    input_data, kernel = generate_random_input(input_size,kernel_size)

    result_torch = pyTorch_conv2D(input_data, kernel, padding, stride)

    layer = joey.Conv2DV2(kernel_size=(kernel_size)
                        ,input_size=(input_size), padding= (padding,padding)
                        ,stride=(stride,stride),generate_code=True, strict_stride_check= False)
    input_numpy = input_data.detach().numpy()
    kernel_numpy = kernel.detach().numpy()
    result_joey = layer.execute(input_numpy,[0]*len(kernel),kernel_numpy )
    if print_results:
        print ("torch",result_torch)

        print ("joey",result_joey)


    print("Do they match", np.allclose(result_joey, result_torch))

    assert(np.allclose(result_joey, result_torch))


@pytest.mark.parametrize("input_size, kernel_size, padding, stride",
[((2,3,5,5,9), (6,1,3,2), 2, 1),((1,3,21,46,50), (6,5,3,7), 10, 3) ])
def test_joey_pytorch_conv3d(input_size, kernel_size, padding, stride, print_results = False):
    ''' test function for 3d conv operation'''
    input_data, kernel = generate_random_input(input_size,kernel_size)

    import time
    start_time = time.time()
    result_torch = pyTorch_conv3D(input_data, kernel, padding, stride)

    print("done with torch,took", time.time() - start_time)
    layer = joey.Conv3D(kernel_size=(kernel_size)
                        ,input_size=(input_size), padding= (padding,padding, padding)
                        ,stride=(stride,stride,stride),generate_code=True, strict_stride_check= False)

    print("start with joey")
    
    result_joey = layer.execute(input_data.detach().numpy(),[0]*len(kernel), kernel.detach().numpy())
   
    print("done with joey, took", time.time() - start_time)

    if print_results:
        print ("torch",result_torch)

        print ("joey",result_joey)


    print("Do they match", np.allclose(result_joey, result_torch))

    assert(np.allclose(result_joey, result_torch))



#test_joey_pytorch_conv3d((5,3,5,9,9), (6,3,3,3), 2, 1, False)

#test_joey_pytorch_conv2d((1,3,7,9), (6,2,3), 2, 1, False)