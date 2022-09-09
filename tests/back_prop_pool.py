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
torch_layer =[]
gstride = 1
gpadding =0
np.set_printoptions(linewidth=np.inf)

torch.manual_seed(0)
class pyTorchModel(nn.Module):
    def __init__(self):
        super(pyTorchModel, self).__init__()
        global torch_conv_op, gstride, gpadding, torch_layer
        self.conv = torch_conv_op
        self.conv1 = torch_layer

    def forward(self, x):
        self.grad_conv =  x = self.conv(x)
        x = self.conv1(x)
        self.grad_conv1 = x
        return x


def pytorch_conv_3d(input_data, kernel_data, padding, stride):
    '''py torch 3d conv'''
    input_size = input_data.size()
    kernel_size = kernel_data.size()
    global torch_conv_op, torch_layer
    torch_conv_op = torch.nn.Conv3d(input_size[1], kernel_size[0],
                                    kernel_size=kernel_size[-1],
                                    padding=padding, stride=stride)

    torch_layer = torch.nn.MaxPool3d(
        kernel_size=kernel_size[2], return_indices=True, padding=padding, stride=stride)


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
    global torch_conv_op, gstride, gpadding, torch_layer
    torch_conv_op = torch.nn.Conv2d(input_size[1], kernel_size[0],
                                    kernel_size=kernel_size[-1],
                                    padding=padding, stride=stride, dtype=torch.double)

    torch_layer = torch.nn.MaxPool2d(
        kernel_size=1, return_indices=True, padding=padding, stride=stride)

    model = pyTorchModel()
    with torch.no_grad():
        # model.conv1.weight = torch.nn.Parameter(kernel_data)
        # model.conv1.bias = torch.nn.Parameter(
        #     torch.Tensor([0]*kernel_size[0]))
        model.conv.weight = torch.nn.Parameter(torch.ones((kernel_size[0],input_size[1], kernel_size[-2],kernel_size[-2]), dtype=torch.double))
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
                         [((2, 3, 5, 5), (6, 2, 3), 5, 1),
                          ((1, 3, 48, 50), (6, 5, 7), 10, 3)])
def test_joey_pytorch_conv2d(input_size, kernel_size, padding, stride,
                             print_results=False):
    ''' test function for 3d conv operation'''
    global gstride,gpadding
    gstride=stride
    gpadding =padding

    input_data, kernel = generate_random_input(input_size, kernel_size)

    pytorch_net = pytorch_conv_2d(input_data, kernel, padding, stride)
   
    layer0 = joey.Conv2DV2(kernel_size=(kernel_size), input_size=(input_size),
                          padding=(padding, padding), stride=(
                              stride, stride), generate_code=True,
                          strict_stride_check=False)

    x = layer0.result.shape

    layer = joey.MaxPooling2D(kernel_size=((1,1)),
                              input_size=(layer0._R.shape), padding=(
                                  padding, padding),
                              stride=(stride, stride),
                              generate_code=True, strict_stride_check=False)

    input_numpy = input_data.detach().numpy()
    kernel_numpy = kernel.detach().numpy()
    
    layers=[layer0,layer]
    joey_net = joey.Net(layers)
    joey_net._layers[0].kernel.data[:] = np.ones((input_size[1],input_size[1], 1,1), dtype=np.double)
    joey_net._layers[0].bias.data[:] = np.array([0]*kernel_size[0])
    criterion = nn.MSELoss()

    pytorch_net.zero_grad()
    outputs = pytorch_net(input_data.double())
    exp_res = torch.randn(outputs[0].shape,dtype=torch.double)
    loss = criterion(outputs[0],exp_res )
    loss.retain_grad()
    def loss_f(pre,label):
        pre = pre.result.data
        N= np.prod(pre.shape)
        res = (2*(pre-label)/N).T
        return res
    out = joey_net.forward(input_numpy)
    joey_net.backward(exp_res.detach().numpy(), loss_f)
    
    result_joey = joey_net._layers[0].result_gradients.data

    from torch.autograd import grad
    result_torch = grad(outputs=loss, inputs=outputs[0], retain_graph= True)[0].detach().numpy()

    print("torch \n", result_torch)

    print(joey_net._layers[1].result_gradients.data)


    print("ind \n", joey_net._layers[1]._indices.data)


    result_torch = grad(outputs=loss, inputs=pytorch_net.grad_conv, retain_graph= True)[0].detach().numpy()


    loss.backward()


    # print(pytorch_net.conv.weight.grad.detach().numpy())
    # print(result_torch1)
    # print(pytorch_net.grad_conv)
    if print_results:
        print("torch ","\n ", result_torch)

        print("joey ","\n" , result_joey)

    print("Do they match", np.allclose(result_joey, result_torch))



# test_joey_pytorch_conv2d((1, 1, 7, 7), (5,5, 5), 2, 1, True)

@pytest.mark.parametrize("input_size, kernel_size, padding, stride",
                         [((2, 3, 5, 5), (6, 2, 3), 5, 1),
                          ((1, 3, 48, 50), (6, 5, 7), 10, 3)])
def test_joey_pytorch_conv3d(input_size, kernel_size, padding, stride,
                             print_results=False):
    global gstride,gpadding
    gstride=stride
    gpadding =padding

    input_data, kernel = generate_random_input(input_size, kernel_size)

    pytorch_net = pytorch_conv_3d(input_data, kernel, padding, stride)
   
    layer0 = joey.Conv3D(kernel_size=(kernel_size), input_size=(input_size),
                          padding=padding, stride=stride, generate_code=True,
                          strict_stride_check=False)

    x = layer0.result.shape

    layer = joey.MaxPooling3D(kernel_size=((kernel_size[1:])),
                              input_size=(layer0._R.shape), padding=padding,
                              stride=stride,
                              generate_code=True, strict_stride_check=False)

    input_numpy = input_data.detach().numpy()
    kernel_numpy = kernel.detach().numpy()
    
    layers=[layer0,layer]
    joey_net = joey.Net(layers)
    print("out_jj \n", joey_net._layers[1]._I.data)

    joey_net._layers[0].kernel.data[:] = kernel
    joey_net._layers[0].bias.data[:] = np.array([0]*kernel_size[0])
    criterion = nn.MSELoss()
    pytorch_net.zero_grad()
    outputs = pytorch_net(input_data.double())
    exp_res = torch.randn(outputs[0].shape,dtype=torch.double)
    loss = criterion(outputs[0],exp_res )
    loss.retain_grad()
    def loss_f(pre,label):
        pre = pre.result.data
        N= np.prod(pre.shape)
        res = (2*(pre-label)/N).T
        return res
    out = joey_net.forward(input_numpy)
    print("yuhu")
    joey_net.backward(exp_res.detach().numpy(), loss_f)

    result_joey = joey_net._layers[0].result_gradients.data

    from torch.autograd import grad
    result_torch = grad(outputs=loss, inputs=outputs[0], retain_graph= True)[0].detach().numpy()

    print("putpt_tt \n", outputs[0])

    print("out_jj \n", joey_net._layers[1].result.data)

    print("out_jj \n", joey_net._layers[0].result.data)
    print("out_jj \n", joey_net._layers[1]._I.data)

    print(joey_net._layers[1].result_gradients.data)

    print("ind \n", result_torch)
    print("ind \n", pytorch_net.grad_conv1[1])

    print("ind \n", joey_net._layers[1]._indices.data)


    result_torch = grad(outputs=loss, inputs=pytorch_net.grad_conv, retain_graph= True)[0].detach().numpy()


    loss.backward()


    # print(pytorch_net.conv.weight.grad.detach().numpy())
    # print(result_torch1)
    print(result_torch)
    if print_results:
        print("torch ","\n ", result_torch)

        print("joey ","\n" , result_joey)

    print("Do they match", np.allclose(result_joey, result_torch))


test_joey_pytorch_conv3d((4,1, 16, 16, 16), (1,5, 5, 5), 2, 3, True)