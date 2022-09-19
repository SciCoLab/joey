import pytest
import numpy as np
from devito import configuration
import joey
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad

configuration['language'] = 'openmp'
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
        # model.conv.weight = torch.nn.Parameter(torch.ones((input_size[1], kernel_size[0],1,1), dtype=torch.double))
        # model.conv.bias = torch.nn.Parameter(
        #     torch.Tensor([0]*kernel_size[0]))

    return model


def pytorch_conv_2d(input_data, kernel_data, padding, stride):
    '''py torch 2d conv'''
    input_size = input_data.size()
    kernel_size = kernel_data.size()
    global torch_conv_op, gstride, gpadding, torch_conv
    torch_conv_op = torch.nn.InstanceNorm2d(input_data.shape[1])

    torch_conv = torch.nn.Conv2d(input_size[1], kernel_size[0],
                                 kernel_size=kernel_size[-1],
                                 padding=gpadding, stride=gstride, dtype=torch.double)

    model = pyTorchModel()
    with torch.no_grad():
        model.conv.weight = torch.nn.Parameter(kernel_data)
        model.conv.bias = torch.nn.Parameter(
            torch.Tensor([0]*kernel_size[0]))
        # model.conv.weight = torch.nn.Parameter(torch.ones((input_size[1], kernel_size[0],1,1), dtype=torch.double))
        # model.conv.bias = torch.nn.Parameter(
        #     torch.Tensor([0]*kernel_size[0]))

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


def conv(input, kernel, padding, stride):
    pass
    # ar_input = torch.zeros((1,1,1,1), dtype=torch.double)
    # ar_input[0][0][0][0] = input[0][0][0][0]

    # input  = ar_input
    # torch_conv_op = torch.nn.Conv2d(input.shape[1], kernel.shape[0],
    #                                 kernel_size=kernel.shape[-1],
    #                                 padding=padding, stride=1, dtype=torch.double)

    # with torch.no_grad():
    #    kernel_size[-1]


def test_joey_pytorch_conv2d(input_size, kernel_size, padding, stride,
                             print_results=False):
    ''' test function for 3d conv operation'''
    global gstride, gpadding
    gstride = stride
    gpadding = padding

    input_data, kernel = generate_random_input(input_size, kernel_size)

    pytorch_net = pytorch_conv_2d(input_data, kernel, padding, stride)

    layer = joey.Conv2DV2(kernel_size, input_size=(input_size),
                          padding=(padding, padding), stride=(
                              stride, stride), generate_code=True,
                          strict_stride_check=False)

    x = (input_size[-1]+(2*padding)-kernel_size[-1])//stride + 1

    layer2 = joey.InstanceNorm2D(input_size=(*input_size[0:2],x,x),
                              generate_code=True)


    input_numpy = input_data.detach().numpy()
    kernel_numpy = kernel.detach().numpy()
    layers = [layer, layer2]
    joey_net = joey.Net(layers)
    joey_net._layers[0].kernel.data[:] = kernel_numpy
    joey_net._layers[0].bias.data[:] = np.array([0]*kernel_size[0])
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
    out = joey_net.forward(input_numpy)
    joey_net.backward(exp_res.detach().numpy(), loss_f)

    result_joey = joey_net._layers[0].kernel_gradients.data
    result_joey = joey_net._layers[0].result_gradients.data
    from torch.autograd import grad

    # temp = torch.from_numpy(joey_net._layers[0].result_gradients.data)

    # result_torch1 = conv(temp,torch.flip(kernel,dims=(2,3)),kernel.shape[-1]-1,stride).detach().numpy()
    # input shape (1,1,7,7)
    result_grad_joey = joey_net._layers[1].result_gradients.data

    input_numpy = joey_net._layers[1]._I.data
    N = np.prod(input_numpy.shape)
    mean = np.sum(input_numpy)/N
    input_mean = input_numpy - mean
    var = np.sum(input_mean*input_mean)/N
    var= var+0.00001
    print(var)
    var_sqrt = np.sqrt(var)
    eq1 = (1-1/N)*var_sqrt
    eq2 = ((result_grad_joey - mean)*(result_grad_joey - mean))/(var_sqrt*N)
    y = (1/var)*(eq1- eq2)
    
    global c, c1
    
    result_torch = grad(outputs=loss, inputs=c, allow_unused=True, retain_graph=True)[0]
   

    loss.backward()

    # result_torch = pytorch_net.conv.weight.grad

    # print(result_torch1)
    
    result_torch = result_torch.detach().numpy()
    if print_results:
        print("torch", result_torch)

        print("joey", result_joey)

    print("Do they match", np.allclose(result_joey, result_torch))

    #assert (np.allclose(result_joey, result_torch))


test_joey_pytorch_conv2d((1, 1, 7, 7), (1, 2, 2), 0, 1, True)


# def test_joey_pytorch_conv3d(input_size, kernel_size, padding, stride,
#                              print_results=False):
#     ''' test function for 3d conv operation'''
#     global torch_conv_op, gstride, gpadding, torch_conv
#     gstride = stride
#     gpadding = padding

#     input_data, kernel = generate_random_input(input_size, kernel_size)

#     pytorch_net = pytorch_conv_3d(input_data, kernel, padding, stride)

#     layer = joey.Conv3D(kernel_size, input_size=(input_size),
#                         padding=padding, stride=stride, generate_code=True,
#         strict_stride_check=False)

#     x = (input_size[-1]+(2*padding)-kernel_size[-1])//stride + 1

#     layer2 = joey.UpSample(input_size=(input_size[0], 1, x, x, x),
#                            scale_factor=kernel_size[-1], generate_code=True)

#     input_numpy = input_data.detach().numpy()
#     kernel_numpy = kernel.detach().numpy()

#     layers = [layer, layer2]
#     joey_net = joey.Net(layers)
#     joey_net._layers[0].kernel.data[:] = kernel_numpy
#     joey_net._layers[0].bias.data[:] = np.array([0]*kernel_size[0])
#     criterion = nn.MSELoss()

#     pytorch_net.zero_grad()
#     outputs = pytorch_net(input_data.double())
#     exp_res = torch.randn(outputs.shape, dtype=torch.double)
#     loss = criterion(outputs, exp_res)
#     loss.retain_grad()

#     def loss_f(pre, label):
#         pre = pre.result.data
#         N = np.prod(pre.shape)
#         res = (2*(pre-label)/N).T
#         return res
#     joey_net.forward(input_numpy)
#     joey_net.backward(exp_res.detach().numpy(), loss_f)

#     result_joey = joey_net._layers[0].kernel_gradients.data
#     # print(joey_net._layerskernel_size = kernel_data.size()

#     # temp = torch.from_numpy(joey_net._layers[0].result_gradients.data)

#     # result_torch1 = conv(temp,torch.flip(kernel,dims=(2,3)),kernel.shape[-1]-1,stride).detach().numpy()
#     global c, c1
#     result_torch = grad(outputs=loss, inputs=c, allow_unused=True,
#                         retain_graph=True)[0].detach().numpy()
#     print(result_torch)

#     loss.backward()

#     result_torch = pytorch_net.conv.weight.grad.detach().numpy()
#        # print(result_torch1)
#     if print_results:
#         print("torch", result_torch)

#         print("joey", result_joey)

#     print("Do they match", np.allclose(result_joey, result_torch))


# test_joey_pytorch_conv3d((4, 3, 5, 5, 5), (1, 3, 3, 3), 0, 1, True)
