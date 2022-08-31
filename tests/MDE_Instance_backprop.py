import pytest
import numpy as np
from devito import configuration
import joey
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad

torch.manual_seed(0)


def generate_random_input(input_size) -> tuple:
    '''generate random data for test'''

    input_data = \
        torch.randn(input_size, dtype=torch.double, requires_grad=True)

    return input_data

'''
grad_res is the incoming DL/DY
'''
def np_backprop_eq(input_numpy, grad_res):
    # computing var & mean using input data : simulating forward_pass
    N = np.prod(input_numpy.shape)
    mean = np.sum(input_numpy)/N
    input_mean = input_numpy - mean
    var = np.sum(input_mean*input_mean)/N
    var= var+0.00001
    var_sqrt = np.sqrt(var)

    # back_prop start
    eq1 = (1-1/N)*var_sqrt
    eq2 = ((grad_res - mean)*(grad_res - mean))/(var_sqrt*N)

    #DL/DX
    y = (1/var)*(eq1- eq2)
    return y

input_data = generate_random_input((1,1,5,5))

criterion = nn.MSELoss()

# Code to compute instance normalization using torch 
# computing var & mean using input data : simulating forward_pass

N = torch.Tensor([25])
mean = torch.sum(input_data)/N
input_mean = input_data - mean
var = torch.sum(input_mean*input_mean)/N
var= var+0.00001
var_sqrt = torch.sqrt(var)
outputs = input_mean/var_sqrt
print(outputs)
# checking correctness with torch instance norm
torch_instance_op = torch.nn.InstanceNorm2d(input_data.shape[1])
print(torch_instance_op(input_data))
exp_res = torch.randn(outputs.shape, dtype=torch.double)
# any loss function for autograd
loss = criterion(outputs,exp_res)
# DL/DY
res_grad = grad(outputs=loss, inputs=outputs, allow_unused=True,
                        retain_graph=True)[0].detach().numpy()

#DL/DX
result_torch = grad(outputs=loss, inputs=input_data, allow_unused=True,
                        retain_graph=True)
   
result_np = np_backprop_eq(input_data.detach().numpy(), res_grad)

print(result_torch)

print(result_np)

