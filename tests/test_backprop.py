from codecs import getreader
import pytest
import numpy as np
from devito import configuration
import joey
import torch
import torch.nn as nn
import torch.nn.functional as F

configuration['language'] = 'openmp'
# configuration['LOGGING'] = 'debug'
torch_conv_op = []
torch.manual_seed(0)
class pyTorchModel(nn.Module):
    def __init__(self):
        super(pyTorchModel, self).__init__()
        global torch_conv_op
        self.conv1 = torch_conv_op

    def forward(self, x):
        
        x = self.conv1(x)
        return x


def pytorch_conv_3d(input_data, kernel_data, padding, stride):
    '''py torch 3d conv'''
    input_size = input_data.size()
    kernel_size = kernel_data.size()
    global torch_conv_op
    torch_conv_op = torch.nn.Conv3d(input_size[1], kernel_size[0],
                                    kernel_size=kernel_size[-1],
                                    padding=padding, stride=stride)

    model = pyTorchModel()
    with torch.no_grad():
        model[0].weight = torch.nn.Parameter(kernel_data)
        model[0].bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]))


    return model


def pytorch_conv_2d(input_data, kernel_data, padding, stride):
    '''py torch 2d conv'''
    input_size = input_data.size()
    kernel_size = kernel_data.size()
    global torch_conv_op
    torch_conv_op = torch.nn.Conv2d(input_size[1], kernel_size[0],
                                    kernel_size=kernel_size[-1],
                                    padding=padding, stride=stride)

    model = pyTorchModel()
    with torch.no_grad():
        model.conv1.weight = torch.nn.Parameter(kernel_data)
        model.conv1.bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]))


    return model


def generate_random_input(input_size, kernel_size) -> tuple:
    '''generate random data for test'''

    kernel = torch.randn(
        kernel_size[0], input_size[1], kernel_size[-2], kernel_size[-1],
        dtype=torch.double)
    input_data = \
        torch.randn(input_size[0], input_size[1],
                    input_size[-2], input_size[-1], dtype=torch.double)
    if len(input_size) == 5:
        kernel = torch.randn(kernel_size[0], input_size[1], kernel_size[-3],
                             kernel_size[-2], kernel_size[-1],
                             dtype=torch.double)
        input_data = \
            torch.randn(input_size[0], input_size[1], input_size[2],
                        input_size[3], input_size[4], dtype=torch.double)

    return input_data, kernel



def convolution2d(image, kernel, bias=0):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) + bias
    return new_image

@pytest.mark.parametrize("input_size, kernel_size, padding, stride",
                         [((2, 3, 5, 5), (6, 2, 3), 5, 1),
                          ((1, 3, 48, 50), (6, 5, 7), 10, 3)])
def test_joey_pytorch_conv2d(input_size, kernel_size, padding, stride,
                             print_results=False):
    ''' test function for 3d conv operation'''
    input_data, kernel = generate_random_input(input_size, kernel_size)

    pytorch_net = pytorch_conv_2d(input_data, kernel, padding, stride)

    layer = joey.Conv2DV2(kernel_size=(kernel_size), input_size=(input_size),
                          padding=(padding, padding), stride=(
                              stride, stride), generate_code=True,
                          strict_stride_check=False)

    input_numpy = input_data.detach().numpy()
    kernel_numpy = kernel.detach().numpy()

    

    joey_net = joey.Net(layers=[layer])
    joey_net._layers[0].kernel.data[:] = kernel_numpy
    joey_net._layers[0].bias.data[:] = np.array([0]*kernel_size[0])
    criterion = nn.MSELoss()

    pytorch_net.zero_grad()
    outputs = pytorch_net(input_data)
    exp_res = torch.randn(outputs.shape,dtype=torch.double)
    loss = criterion(outputs,exp_res )
    loss.retain_grad()
    loss.backward()
    def loss_f(pre,label):
        pre = pre.result.data
        N= np.prod(pre.shape)
        return (2*(pre-label)/N).T
    joey_net.forward(input_numpy)
    print(loss.grad)
    joey_net.backward(exp_res.detach().numpy(), loss_f)
    
    result_joey = joey_net._layers[0].kernel_gradients.data

  
    result_torch = pytorch_net.conv1.weight.grad.detach().numpy()
    if print_results:
        print("torch", result_torch)

        print("joey", result_joey)

    print("Do they match", np.allclose(result_joey, result_torch))

    #assert (np.allclose(result_joey, result_torch))


test_joey_pytorch_conv2d((5, 1, 5, 5), (1, 3, 3), 0, 1, True)