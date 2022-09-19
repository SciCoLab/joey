import numpy as np
from devito import configuration
import joey
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from diceloss import DiceLoss
configuration['language'] = 'openmp'
#configuration['opt'] = 'advanced'
#configuration['platform']='nvidiaX'
torch_layer = []
torch_conv = []
gstride = 1
gpadding = 0
c = c1 = []
torch.manual_seed(0)

np.set_printoptions(linewidth=1000)
class pyTorchModel(nn.Module):
    def __init__(self):
        super(pyTorchModel, self).__init__()
        global torch_layer, gstride, gpadding, torch_conv
        self.conv = torch_conv
        self.layer = torch_layer

    def forward(self, x):

        x = self.conv(x)
        global c, c1
        c = x
        c1 = x = self.layer(x)
        x = F.relu(x)
        return x


def pytorch_conv_3d(input_data, kernel_data, padding, stride):
    '''py torch 3d conv'''
    input_size = input_data.size()
    kernel_size = kernel_data.size()
    global torch_layer, gstride, gpadding, torch_conv
    torch_layer = nn.InstanceNorm3d(input_size[1], dtype=torch.double)

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
    global torch_layer, gstride, gpadding, torch_conv
    torch_layer = nn.InstanceNorm2d(input_size[1],dtype=torch.double)

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
        input_data = \
        torch.randn(input_size[0], input_size[1],
                    input_size[-3],input_size[-2], input_size[-1], dtype=torch.double, requires_grad=True)
 
        kernel = torch.randn(kernel_size[0], input_size[1], kernel_size[-3],
                             kernel_size[-2],kernel_size[-1],dtype=torch.double)  
    return input_data, kernel


def test_joey_pytorch_conv2d(input_size, kernel_size, padding, stride,
                             print_results=False):
    ''' test function for 3d conv operation'''
    global gstride, gpadding
    gstride = stride
    gpadding = padding
    criterion = nn.MSELoss()

    input_data, kernel = generate_random_input(input_size, kernel_size)

    pytorch_net = pytorch_conv_2d(input_data, kernel, padding, stride)

    layer = joey.Conv2DV2(kernel_size, input_size=(input_size),
                          padding=(padding, padding), stride=(
                              stride, stride), generate_code=True,
                          strict_stride_check=False)

    x = layer.result.shape
    # activation=joey.activation.ReLU()
    layer2 = joey.InstanceNorm2D( input_size= x , generate_code=True, activation=joey.activation.ReLU())

    input_numpy = input_data.detach().numpy()
    kernel_numpy = kernel.detach().numpy()

    layers = [layer, layer2]
    joey_net = joey.Net(layers)
    joey_net._layers[0].kernel.data[:] = kernel_numpy
    joey_net._layers[0].bias.data[:] = np.array([0]*kernel_size[0])
    pytorch_net.zero_grad()
    outputs = pytorch_net(input_data.double())
    exp_res = torch.randn(outputs.shape, dtype=torch.double)
    loss = criterion(outputs, exp_res)
    loss.retain_grad()

    def loss_f(pre, label):
        pre = pre.result.data
        #pre = torch.sigmoid(torch.from_numpy(pre)).detach().numpy()    
        un = np.sum(pre**2)+np.sum(label**2)
        num =  (label * un) - (2*pre*np.sum(pre*label))
        dem =   (un)**2 
        res = -2*(num/dem)
        # label = np.abs(np.sign(pre))* label
        N = np.prod(pre.shape)
        res = (2*(pre-label)/N)
        return res.T
    joey_net.forward(input_numpy)
    joey_net.backward(exp_res.detach().numpy(), loss_f)

    result_joey = joey_net._layers[0].kernel_gradients.data
    global c, c1
    # result_joey = (joey_net._layers[0].result_gradients.data)
    input_numpy = joey_net._layers[1]._I.data
    N = np.prod(input_numpy.shape[2:])
    mean = np.sum(input_numpy,axis=(2,3))/N
    input_mean = input_numpy - mean[:,:,None,None]
    var = np.sum(input_mean*input_mean, axis=(2,3))/N
    var= var+0.00001
    var_sqrt = np.sqrt(var)
    inv_variance = 1/var_sqrt
    grad_res = joey_net._layers[1].result_gradients.data
    output = c1.detach().numpy()
    # N = np.count_nonzero(joey_net._layers[1]._R.data)
    e1 =  (N)*(grad_res) 
    e2 = np.sum(grad_res)
    e3 = output * np.sum(grad_res*output)
    e4 = e1-e2-e3
    y1 = e4 * inv_variance[:,:,None,None] * (1/N)
    print(y1)
    from torch.autograd import grad


    result_torch = grad(outputs=loss, inputs=c, allow_unused=True,
                        retain_graph=True)[0].detach().numpy()

    loss.backward()
    
    result_torch = pytorch_net.conv.weight.grad.detach().numpy()
    # print(result_torch1)
    if print_results:
        print("torch \n", result_torch)

        print("joey \n", result_joey)

    print("Do they match", np.allclose(result_joey, result_torch))
    # print("diff ", np.subtract(result_joey, result_torch))
    #assert (np.allclose(result_joey, result_torch))


#test_joey_pytorch_conv2d((2, 1, 7, 7), (1, 3, 3), 0, 1, True)


def test_joey_pytorch_conv3d(input_size, kernel_size, padding, stride,
                             print_results=False):
    ''' test function for 3d conv operation'''
    global torch_layer, gstride, gpadding, torch_conv
    gstride = stride
    gpadding = padding

    input_data, kernel = generate_random_input(input_size, kernel_size)

    pytorch_net = pytorch_conv_3d(input_data, kernel, padding, stride)

    layer = joey.Conv3D(kernel_size, input_size=(input_size),
                        padding=padding, stride=stride, generate_code=True,
        strict_stride_check=False)

    x = layer.result.shape
    import time
    start = time.time()

    layer2 = joey.InstanceNorm3D( input_size= x , generate_code=True, activation=joey.activation.ReLU())

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
    joey_net.forward(input_numpy)
    joey_net.backward(exp_res.detach().numpy(), loss_f)

    result_joey = joey_net._layers[0].kernel_gradients.data
    #result_joey = joey_net._layers[1].kernel_gradients.data
    global c, c1
    result_torch = grad(outputs=loss, inputs=c, allow_unused=True,
                        retain_graph=True)[0].detach().numpy()

    loss.backward()

    result_torch = pytorch_net.conv.weight.grad.detach().numpy()
       # print(result_torch1)
    if print_results:
        print("torch", result_torch)

        print("joey", result_joey)

    print("Do they match", np.allclose(result_joey, result_torch))
    print("lalala", start - time.time())


test_joey_pytorch_conv3d((4, 4, 16, 16, 16), (32, 3, 3, 3), 0, 1, False)
