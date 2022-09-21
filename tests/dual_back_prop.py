from codecs import getreader
import pytest
import numpy as np
from devito import configuration
import joey
import torch
import torch.nn as nn
import torch.nn.functional as F

from joey import activation

configuration['compiler'] = 'nvc'
configuration['language'] = 'openmp'
configuration['opt'] = 'advanced'
torch_conv_op = []
torch_conv = []
gstride = 1
gpadding = 0
torch.manual_seed(0)

np.set_printoptions(linewidth=1000)

class pyTorchModel(nn.Module):
    def __init__(self):
        super(pyTorchModel, self).__init__()
        global torch_conv_op, gstride, gpadding, torch_conv
        self.conv = torch_conv
        self.conv1 = torch_conv_op

    def forward(self, x):
        self.grad_conv = x = self.conv(x)
        
        x  = self.conv1(x)
        x = F.relu(x)
        return x


def pytorch_conv_3d(input_data, kernel_data, padding, stride):
    '''py torch 3d conv'''
    input_size = input_data.size()
    kernel_size = kernel_data.size()
    global torch_conv_op, gstride, gpadding, torch_conv
    torch_conv_op = torch.nn.Conv3d(input_size[1], kernel_size[0],
                                    kernel_size=kernel_size[-1],
                                    padding=padding, stride=stride, dtype=torch.double)

    torch_conv = torch.nn.Conv3d(input_size[1], kernel_size[0],
                                 kernel_size=1,
                                 padding=gpadding, stride=gstride, dtype=torch.double)

    model = pyTorchModel()
    with torch.no_grad():
        model.conv1.weight = torch.nn.Parameter(kernel_data)
        model.conv1.bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]))
        model.conv.weight = torch.nn.Parameter(torch.ones(
            (kernel_size[0], input_size[1], 1,1, 1), dtype=torch.double))
        model.conv.bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]))

    return model

def pytorch_conv_2d(input_data, kernel_data, padding, stride):
    '''py torch 2d conv'''
    input_size = input_data.size()
    kernel_size = kernel_data.size()
    global torch_conv_op, gstride, gpadding, torch_conv
    torch_conv_op = torch.nn.Conv2d(input_size[1], kernel_size[0],
                                    kernel_size=kernel_size[-1],
                                    padding=padding, stride=stride, dtype=torch.double)

    torch_conv = torch.nn.Conv2d(input_size[1], kernel_size[0],
                                 kernel_size=1,
                                 padding=gpadding, stride=stride, dtype=torch.double)

    model = pyTorchModel()
    with torch.no_grad():
        model.conv1.weight = torch.nn.Parameter(kernel_data)
        model.conv1.bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]))
        model.conv.weight = torch.nn.Parameter(torch.ones(
            (kernel_size[0], input_size[1], 1, 1), dtype=torch.double))
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


@pytest.mark.parametrize("input_size, kernel_size, padding, stride",
                         [((3, 3, 5, 5), (3, 3, 3), 2, 2),
                          ((1, 3, 50, 50), (3, 5, 5), 2, 1)])
def test_joey_pytorch_conv2d(input_size, kernel_size, padding, stride,
                             print_results=False):
    ''' test function for 2d conv operation'''
    global gstride, gpadding
    gstride = stride
    gpadding = padding

    input_data, kernel = generate_random_input(input_size, kernel_size)

    pytorch_net = pytorch_conv_2d(input_data, kernel, padding, stride)

    layer0 = joey.Conv2DV2(kernel_size=(kernel_size[0], 1, 1), input_size=(input_size),
                           padding=(padding, padding), stride=(
        stride, stride),  generate_code=True,
        strict_stride_check=False)

    x = layer0.result.shape

    layer = joey.Conv2DV2(kernel_size=(kernel_size), input_size=(x),
                          padding=padding, stride = stride, generate_code=True,
                          activation=joey.activation.ReLU(), strict_stride_check=False)

    input_numpy = input_data.detach().numpy()
    kernel_numpy = kernel.detach().numpy()

    layers = [layer0, layer]
    joey_net = joey.Net(layers)
    joey_net._layers[0].kernel.data[:] = np.ones(
        (kernel_size[0], input_size[1], 1, 1), dtype=np.double)
    joey_net._layers[0].bias.data[:] = np.array([0]*kernel_size[0])
    joey_net._layers[1].kernel.data[:] = kernel_numpy
    joey_net._layers[1].bias.data[:] = np.array([0]*kernel_size[0])
    criterion = nn.MSELoss()

    pytorch_net.zero_grad()
    outputs = pytorch_net(input_data.double())
    exp_res = torch.randn(outputs.shape, dtype=torch.double)
    loss = criterion(outputs, exp_res)
    loss.retain_grad()

    def loss_f(pre, label):
        pre = pre.result.data
        N = np.prod(pre.shape)
        res = (2*(pre-label)/N).T
        return res
    joey_net.forward(input_numpy)
    joey_net.backward(exp_res.detach().numpy(), loss_f)

    result_joey = joey_net._layers[0].result_gradients.data
    result_joey = joey_net._layers[0].kernel_gradients.data

    from torch.autograd import grad

    result_torch = grad(outputs=loss, inputs=outputs,
                        retain_graph=True)[0].detach().numpy()
    print(result_torch)
    print(joey_net._layers[1].result_gradients.data)
    loss.backward()
    result_torch = pytorch_net.conv.weight.grad.detach().numpy()

    print(joey_net._layers[1].result.data)

    if print_results:
        print("torch", result_torch)

        print("joey", result_joey)

        print("Do they match", np.allclose(result_joey, result_torch))

   # assert (np.allclose(result_joey, result_torch))


test_joey_pytorch_conv2d((1, 50, 500, 500), (50, 3, 3), 2, 1, True)


@pytest.mark.parametrize("input_size, kernel_size, padding, stride",
                         [((2, 3, 9, 9, 9), (3, 3, 3, 3), 2, 1),
                          ((1, 3, 21, 46, 50), (3, 5, 5, 5), 10, 3)])
def test_joey_pytorch_conv3d(input_size, kernel_size, padding, stride,
                             print_results=False):
    ''' test function for 3d conv operation'''
    global gstride, gpadding
    gstride = stride
    gpadding = padding

    input_data, kernel = generate_random_input(input_size, kernel_size)

    pytorch_net = pytorch_conv_3d(input_data, kernel, padding, stride)

    layer0 = joey.Conv3D(kernel_size=(kernel_size[0],1, 1, 1), input_size=(input_size),
                           padding=padding, stride=stride, activation=joey.activation.ReLU(), generate_code=True,
        strict_stride_check=False)

    x = layer0.result.shape

    layer = joey.Conv3D(kernel_size=(kernel_size), input_size=(x),
                          padding=padding, stride=stride, generate_code=True,
                          strict_stride_check=False)

    input_numpy = input_data.detach().numpy()
    kernel_numpy = kernel.detach().numpy()

    layers = [layer0, layer]
    joey_net = joey.Net(layers)
    joey_net._layers[0].kernel.data[:] = np.ones(
        (kernel_size[0], input_size[1], 1, 1, 1), dtype=np.double)
    joey_net._layers[0].bias.data[:] = np.array([0]*kernel_size[0])
    joey_net._layers[1].kernel.data[:] = kernel_numpy
    joey_net._layers[1].bias.data[:] = np.array([0]*kernel_size[0])
    criterion = nn.MSELoss()

    pytorch_net.zero_grad()
    outputs = pytorch_net(input_data.double())
    exp_res = torch.randn(outputs.shape, dtype=torch.double)
    loss = criterion(outputs, exp_res)
    loss.retain_grad()

    def loss_f(pre, label):
        pre = pre.result.data
        N = np.prod(pre.shape)
        res = (2*(pre-label)/N).T
        return res
    joey_net.forward(input_numpy)
    joey_net.backward(exp_res.detach().numpy(), loss_f)
    
    result_joey = joey_net._layers[0].result_gradients.data

    # result_joey = joey_net._layers[0].kernel_gradients.data
    from torch.autograd import grad

    result_torch = grad(outputs=loss, inputs=pytorch_net.grad_conv,
                        retain_graph=True)[0].detach().numpy()
    loss.backward()

    # result_torch = pytorch_net.conv.weight.grad.detach().numpy()
    # print(result_torch1)
    if print_results:
        print("torch", result_torch)

        print("joey", result_joey)

    assert (np.allclose(result_joey, result_torch))

#test_joey_pytorch_conv3d((1,1, 7, 7, 7), (1,3, 3, 3), 0, 1, True)
