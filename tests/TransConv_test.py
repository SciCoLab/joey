"""
gc2016: the file and its version control history can be found here:
https://github.com/cha-tzi/devito/blob/ml/examples/ml/Test_TransConv.py
This is a script that tests the created Transposed convolution function 
against the PyTorch implementation
A filter tensor with weights was initialised and passed as inputdata to the Devito operator.
The same filter was converted to a pytorch param-eter and passed as a .weight parameter 
to the pytorch operator.  Their outputwas compared with the np.allclose function 
and the absolute error was calculated

"""
# =======SUCCESS=================
import torch
import joey as ml
from devito import Operator
import sympy
import numpy as np
# initialisation of a random tensor
input_data = torch.rand((2,4,5,5))
# initialisation of weights
w1 = torch.ones((4, 2, 2, 2))
for i in range(4):
    w1[i] = w1[i]+i+1
# creating the PyTorch operator
torch_trans_noset = torch.nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2)
# Arguments: Number of channels in the input image, Number of channels produced by the convolution

# converting the weights to PyTorch Parameters
torch_trans_noset.weight = torch.nn.Parameter(w1)
# convert the weights back to numpy so they can be used 
# in devito
weight_dev = torch_trans_noset.weight.transpose(0,1).detach().numpy()
# converting the weights to PyTorch Parameters
torch_trans_noset.bias = torch.nn.Parameter(torch.zeros((2,)))
# convert the weights back to numpy so they can be used 
# in devito
bias_dev = torch_trans_noset.bias.detach().numpy()
# pass the data to the PyTorch operator
torch_res_trans = torch_trans_noset(input_data)
# Create the Joey Operator
layer1 = ml.TransConv(kernel_size=(2, 2, 2),
                       input_size=(2,4,5,5),
                       stride = (2,2))
# pass the data to the Joey operator
current_data = layer1.execute(input_data,
        bias_dev, weight_dev ) 
print("Same result:", np.allclose(torch_res_trans.detach().numpy(), current_data))
print("Absolute error :", np.sum(abs(abs(torch_res_trans.detach().numpy()) - abs(current_data))))
