import pytest
from pyTorch_2Dconv import pyTorch_conv
from Devito_Conv_2d import *
import numpy as np

c=0;
input =[]
for i in range(0,5):
    temp =[]
    for j in range(0,5):
        temp.append(c)
        c = c+1;
    input.append(temp)

custom_kernel = [[0,0,0] ,[1,0,-1],[0,0,0]]

#@pytest.mark.parametrize("input, kernel, padding, stride", [(input, custom_kernel, 1, 1),(input, custom_kernel, 2, 1) ])
def compare_Devito_pyTorch(input, kernel, padding, stride):

    result_devito = conv_devito_2D_defaultStride(input, custom_kernel, stride, padding)


    result_torch = pyTorch_conv(input, custom_kernel, stride, padding)

    print (result_devito)

    print (result_torch)

    print(np.allclose(result_devito, result_torch))


compare_Devito_pyTorch(input,custom_kernel,padding=2,stride=1)