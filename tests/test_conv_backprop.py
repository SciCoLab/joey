import pytest
import numpy as np
from devito import configuration
import joey
import torch
import torch.nn as nn
import torch.nn.functional as F

from joey import activation

configuration['compiler'] = 'gcc'
configuration['language'] = 'C'
configuration['opt'] = 'advanced'
torch_conv1 = []
torch_conv0 = []
torch.manual_seed(0)

np.set_printoptions(linewidth=1000)


class pyTorchModel(nn.Module):
    def __init__(self):
        super(pyTorchModel, self).__init__()
        global torch_conv1, torch_conv0
        self.conv = torch_conv0
        self.conv1 = torch_conv1

    def forward(self, x):

        self.grad_conv = x = self.conv(x)

        x = self.conv1(x)
        self.grad_conv1 = x = F.relu(x)
        return x


def pytorch_conv_3d(input_data, kernel_data, padding, stride):
    '''py torch 3d conv'''
    input_size = input_data.size()
    kernel_size = kernel_data.size()
    global torch_conv1, torch_conv0
    torch_conv1 = torch.nn.Conv3d(input_size[1], kernel_size[0],
                                  kernel_size=kernel_size[-1],
                                  padding=padding, stride=stride,
                                  dtype=torch.double)

    torch_conv0 = torch.nn.Conv3d(input_size[1], kernel_size[0],
                                  kernel_size=kernel_size[-1],
                                  padding=padding, stride=stride,
                                  dtype=torch.double)

    model = pyTorchModel()
    with torch.no_grad():
        model.conv1.weight = torch.nn.Parameter(kernel_data)
        model.conv1.bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]))
        model.conv.weight = torch.nn.Parameter(kernel_data)
        model.conv.bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]))

    return model


def pytorch_conv_2d(input_data, kernel_data, padding, stride):
    '''py torch 2d conv'''
    input_size = input_data.size()
    kernel_size = kernel_data.size()
    global torch_conv1, gstride, gpadding, torch_conv0
    torch_conv1 = torch.nn.Conv2d(input_size[1], kernel_size[0],
                                  kernel_size=kernel_size[-1],
                                  padding=padding, stride=stride,
                                  dtype=torch.double)

    torch_conv0 = torch.nn.Conv2d(input_size[1], kernel_size[0],
                                  kernel_size=kernel_size[-1],
                                  padding=padding, stride=stride,
                                  dtype=torch.double)

    model = pyTorchModel()
    with torch.no_grad():
        model.conv1.weight = torch.nn.Parameter(kernel_data)
        model.conv1.bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]).double())
        model.conv.weight = torch.nn.Parameter(kernel_data)
        model.conv.bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]).double())

    return model


def generate_random_input(input_size, kernel_size) -> tuple:
    '''generate random data for test'''

    kernel = torch.randn(
        kernel_size[0], input_size[1], kernel_size[-2], kernel_size[-1],
        dtype=torch.double)
    input_data = \
        torch.randn(input_size[0], input_size[1],
                    input_size[-2], input_size[-1], dtype=torch.double,
                    requires_grad=True)
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
                          ((1, 3, 45, 50), (3, 5, 5), 2, 1)])
def test_joey_pytorch_conv2d(input_size, kernel_size, padding, stride,
                             print_results=False):
    ''' test function for 2d conv operation'''

    input_data, kernel = generate_random_input(input_size, kernel_size)

    pytorch_net = pytorch_conv_2d(input_data, kernel, padding, stride)

    layer0 = joey.Conv2DV2(kernel_size=(kernel_size), input_size=(input_size),
                           padding=(padding, padding), stride=(
        stride, stride),  generate_code=False,
        strict_stride_check=False)

    layer = joey.Conv2DV2(kernel_size=(kernel_size),
                          input_size=layer0.result.shape,
                          padding=padding, stride=stride, generate_code=False,
                          activation=activation.ReLU(),
                          strict_stride_check=False)

    input_numpy = input_data.detach().numpy()
    kernel_numpy = kernel.detach().numpy()

    layers = [layer0, layer]
    joey_net = joey.Net(layers)
    joey_net._layers[0].kernel.data[:] = kernel_numpy
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

    loss.backward()
    grad_torch_W1 = pytorch_net.conv1.weight.grad.detach().numpy()
    grad_torch_b1 = pytorch_net.conv1.bias.grad.detach().numpy()

    grad_torch_W0 = pytorch_net.conv.weight.grad.detach().numpy()
    grad_torch_b0 = pytorch_net.conv.bias.grad.detach().numpy()

    assert np.allclose(grad_torch_W1,
                       joey_net._layers[1].kernel_gradients.data)
    assert np.allclose(grad_torch_b1,
                       joey_net._layers[1].bias_gradients.data)

    assert np.allclose(grad_torch_W0,
                       joey_net._layers[0].kernel_gradients.data)
    assert np.allclose(grad_torch_b0,
                       joey_net._layers[0].bias_gradients.data)


@pytest.mark.parametrize("input_size, kernel_size, padding, stride",
                         [((2, 3, 9, 9, 9), (3, 3, 3, 3), 2, 1),
                          ((1, 3, 21, 21, 21), (3, 5, 5, 5), 5, 3)])
def test_joey_pytorch_conv3d(input_size, kernel_size, padding, stride,
                             print_results=False):
    ''' test function for 3d conv operation'''

    input_data, kernel = generate_random_input(input_size, kernel_size)

    pytorch_net = pytorch_conv_3d(input_data, kernel, padding, stride)

    layer0 = joey.Conv3D(kernel_size=(kernel_size), input_size=(input_size),
                         padding=padding, stride=stride, generate_code=False,
                         strict_stride_check=False)

    layer = joey.Conv3D(kernel_size=(kernel_size),
                        input_size=(layer0.result.shape),
                        padding=padding, stride=stride,
                        activation=activation.ReLU(),
                        generate_code=False, strict_stride_check=False)

    input_numpy = input_data.detach().numpy()
    kernel_numpy = kernel.detach().numpy()

    layers = [layer0, layer]
    joey_net = joey.Net(layers)
    joey_net._layers[0].kernel.data[:] = kernel_numpy
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

    loss.backward()
    grad_torch_W1 = pytorch_net.conv1.weight.grad.detach().numpy()
    grad_torch_b1 = pytorch_net.conv1.bias.grad.detach().numpy()

    grad_torch_W0 = pytorch_net.conv.weight.grad.detach().numpy()
    grad_torch_b0 = pytorch_net.conv.bias.grad.detach().numpy()

    assert np.allclose(grad_torch_W1,
                       joey_net._layers[1].kernel_gradients.data)
    assert np.allclose(grad_torch_b1,
                       joey_net._layers[1].bias_gradients.data)

    assert np.allclose(grad_torch_W0,
                       joey_net._layers[0].kernel_gradients.data)
    assert np.allclose(grad_torch_b0,
                       joey_net._layers[0].bias_gradients.data)
