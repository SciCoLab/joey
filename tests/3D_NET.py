from tkinter import E
import pytest
import numpy as np
from devito import configuration
import joey
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad

configuration['language'] = 'openmp'
configuration['platform']='nvidiaX'
configuration['opt'] ='advanced'
# configuration['LOGGING'] = 'debug'
torch_conv_op = []
torch_conv = []
gstride = 1
gpadding = 0
c = c1 = []
torch.manual_seed(0)


class pyTorchModel(nn.Module):
    def __init__(self):
        super(pyTorchModel, self).__init__()
        global torch_conv_op, gstride, gpadding, torch_conv
        self.conv = torch_conv
        self.conv1 = torch_conv_op

    def forward(self, x):

        x = self.conv(x)
        global c, c1
        c = x
        c1 = x = self.conv1(x)
        return x


def pytorch_conv_3d(input_data, kernel_data, padding, stride):
    '''py torch 3d conv'''
    input_size = input_data.size()
    kernel_size = kernel_data.size()
    global torch_conv_op, gstride, gpadding, torch_conv
    torch_conv_op = nn.Upsample(scale_factor=kernel_size[-1], mode='nearest')

    torch_conv = torch.nn.Conv3d(input_size[1], kernel_size[0],
                                 kernel_size=kernel_size[-1],
                                 padding=gpadding, stride=gstride, dtype=torch.double)

    model = pyTorchModel()
    with torch.no_grad():
        model.conv.weight = torch.nn.Parameter(kernel_data)
        model.conv.bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]))

    return model


def pytorch_conv_2d(input_data, kernel_data, padding, stride):
    '''py torch 2d conv'''
    input_size = input_data.size()
    kernel_size = kernel_data.size()
    global torch_conv_op, gstride, gpadding, torch_conv
    torch_conv_op = nn.Upsample(scale_factor=kernel_size[-1], mode='nearest')

    torch_conv = torch.nn.Conv2d(input_size[1], kernel_size[0],
                                 kernel_size=kernel_size[-1],
                                 padding=gpadding, stride=gstride, dtype=torch.double)

    model = pyTorchModel()
    with torch.no_grad():
        model.conv.weight = torch.nn.Parameter(kernel_data)
        model.conv.bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]))

    return model


def generate_random_input(input_size, kernel_size) -> tuple:
    '''generate random data for test'''

    kernel = torch.randn(
        kernel_size[0], input_size[1], kernel_size[-2], kernel_size[-1],
        dtype=torch.double)
    input_data = \
        torch.randn(input_size[0], input_size[1],
                    input_size[-2], input_size[-1], dtype=torch.double, requires_grad=True)
    if len(input_size) == 5:
        kernel = torch.randn(kernel_size[0], input_size[1], kernel_size[-3],
                             kernel_size[-2], kernel_size[-1],
                             dtype=torch.double)
        input_data = \
            torch.randn(input_size[0], input_size[1], input_size[2],
                        input_size[3], input_size[4], dtype=torch.double)

    return input_data, kernel

def test_joey_pytorch_conv3d(input_size, kernel_size, padding, stride,epoch =10,
                             print_results=False):
    ''' test function for 3d conv operation'''
    global torch_conv_op, gstride, gpadding, torch_conv
    gstride = stride
    gpadding = padding

    input_data, kernel = generate_random_input(input_size, kernel_size)

    pytorch_net = pytorch_conv_3d(input_data, kernel, padding, stride)

    layer = joey.Conv3D(kernel_size, input_size=(input_size),
                        padding=padding, stride=stride, generate_code=True,
        strict_stride_check=False)

    x = layer._R.shape

    layer2 = joey.UpSample(input_size=x,
                           scale_factor=kernel_size[-1], generate_code=True)

    input_numpy = input_data.detach().numpy()
    kernel_numpy = kernel.detach().numpy()

    layers = [layer, layer2]
    joey_net = joey.Net(layers)
    joey_net._layers[0].kernel.data[:] = kernel_numpy
    joey_net._layers[0].bias.data[:] = np.array([0]*kernel_size[0])
    criterion = nn.MSELoss()

    pytorch_net.zero_grad()
    X = input_data.to('cuda')
    outputs = pytorch_net(input_data.double())
    exp_res = torch.randn(outputs.shape, dtype=torch.double)
    import time
    start = time.time()
    datatset = 20
    joey_time_epoch = []
    pyTorch_epoch_time = []
    e_start = time.time()
    for i in range(0,epoch):
        for i in range(0,20):
            outputs = pytorch_net(input_data.double())
            exp_res = torch.randn(outputs.shape, dtype=torch.double)
            loss = criterion(outputs, exp_res)
        pyTorch_epoch_time.append(float(time.time()- e_start))
    print("Time taken by pyTorch", abs(start-time.time()))
    def loss_f(pre, label):
        pre = pre.result.data
        N = np.prod(pre.shape)
        res = (2*(pre-label)/N).T
        return res
    start = time.time()
    e_start = time.time()
    for i in range(0,epoch):

        for i in range(0,20):
            joey_net.forward(input_numpy)
            joey_net.backward(exp_res.detach().numpy(), loss_f)
        joey_time_epoch.append(float(time.time()- e_start))
    print("Time taken by Joey", abs(start-time.time()) )

    result_joey = joey_net._layers[0].kernel_gradients.data
    global c, c1
    result_torch = grad(outputs=loss, inputs=c, allow_unused=True,
                        retain_graph=True)[0].detach().numpy()

    loss.backward()

    result_torch = pytorch_net.conv.weight.grad.detach().numpy()

    print("Do they match", np.allclose(result_joey, result_torch))
    import matplotlib.pyplot as plt
    print(np.max(result_joey-result_torch))
    plt.plot(range(0,epoch), pyTorch_epoch_time, label = "Time per epoch pyTorch")
    plt.plot(range(0,epoch), joey_time_epoch, label = "Time per epoch Joey")

    plt.legend()
    plt.show()
    plt.savefig('foo.png')

test_joey_pytorch_conv3d((2, 1, 8, 32, 32), (12, 3, 3, 3), 2, 2,25, False)
