import pytest
from pyTorch_2Dconv import pyTorch_conv
from Devito_Conv_2d import *
import numpy as np

custom_kernel = [[0.5,0.5,0.5] ,[1,0,-1],[0.5,0.5,0.5]]



@pytest.mark.parametrize("input_size, kernel, padding", [(5, custom_kernel, 2),(50, custom_kernel, 6) ])
def test_Devito_pyTorch_padding(input_size, kernel, padding ):
    c=0;
    input =[]
    for i in range(0,input_size):
        temp =[]
        for j in range(0,input_size):
            temp.append(c)
            c = c+1;
        input.append(temp)

    result_devito = conv_devito_2D(input, kernel, 1, padding)


    result_torch = pyTorch_conv(input, kernel, 1, padding)

    #print ("devito",result_devito)

    #print ("torch",result_torch[0][0])

    assert(np.allclose(result_devito, result_torch))




@pytest.mark.parametrize("input_size, kernel, stride", [(5, custom_kernel, 2),(50, custom_kernel, 5) ])
def test_Devito_pyTorch_stride(input_size, kernel, stride ):
    c=0;
    input =[]
    for i in range(0,input_size):
        temp =[]
        for j in range(0,input_size):
            temp.append(c)
            c = c+1;
        input.append(temp)

    result_devito = conv_devito_2D(input, kernel, stride, 1)


    result_torch = pyTorch_conv(input, kernel, stride, 1)

    #print ("devito",result_devito)

    #print ("torch",result_torch[0][0])

    assert(np.allclose(result_devito, result_torch))





@pytest.mark.parametrize("input_size, kernel, padding, stride", [(5, custom_kernel, 1, 1),(50, custom_kernel, 6, 4) ])
def test_Devito_pyTorch(input_size, kernel, padding, stride):
    c=0;
    input =[]
    for i in range(0,input_size):
        temp =[]
        for j in range(0,input_size):
            temp.append(c)
            c = c+1;
        input.append(temp)

    result_devito = conv_devito_2D(input, kernel, stride, padding)


    result_torch = pyTorch_conv(input, kernel, stride, padding)

    #print ("devito",result_devito)

    #print ("torch",result_torch[0][0])

    assert(np.allclose(result_devito, result_torch))


