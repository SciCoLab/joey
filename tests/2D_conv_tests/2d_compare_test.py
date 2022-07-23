import pytest
from pyTorch_2Dconv import pyTorch_conv
from Devito_Conv_2d import conv_devito_2D,conv_devito_2D_channels
import numpy as np

custom_kernel = [[0,0,0] ,[1,0,-1],[0,0,0]]

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

    result_devito = conv_devito_2D(input, kernel, padding,1)
    kernel = [kernel]
    input =[input]

    result_torch = pyTorch_conv(input, kernel, padding,1)

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

    result_devito = conv_devito_2D(input, kernel, 1, stride)
    kernel = [kernel]
    input =[input]

    result_torch = pyTorch_conv(input, kernel, 1, stride)

    #print ("devito",result_devito)

    #print ("torch",result_torch[0][0])

    assert(np.allclose(result_devito, result_torch))




custom_kernel1 = [[0.5,0.5,0.5,0.5,0.5] ,[1,0,-1,0.5,0.5],[0.5,0.5,0.5,0.5,0.5],[1,0,-1,0.5,0.5],[0.5,0.5,0.5,0.5,0.5]]

@pytest.mark.parametrize("input_size, kernel, padding, stride", [(5, custom_kernel, 1, 1),(50, custom_kernel1, 6, 4) ])
def test_Devito_pyTorch(input_size, kernel, padding, stride):
    c=0;
    input =[]
    for i in range(0,input_size):
        temp =[]
        for j in range(0,input_size):
            temp.append(c)
            c = c+1;
        input.append(temp)
        
    print("padding is ", padding)
    print("stride is ", stride)


    result_devito = conv_devito_2D(input, kernel, padding, stride)

    kernel = [kernel]
    input =[input]
    result_torch = pyTorch_conv(input, kernel, padding, stride)

    print ("devito \n",result_devito,"\n")

    print ("torch \n",result_torch[0][0])

    print("Do they match", np.allclose(result_devito, result_torch))

    assert(np.allclose(result_devito, result_torch))




@pytest.mark.parametrize("input_size, kernel, channels, padding, stride", [(5, custom_kernel, 1, 1, 1),(50, custom_kernel1, 1, 6, 4) ])
def test_Devito_pyTorch_channels(input_size,kernel, channels, padding, stride):
    c=1;
    input =[]
    for i in range(0,input_size):
        temp =[]
        for j in range(0,input_size):
            temp.append(c)
            c = c+1;
        input.append(temp)   
    
    channel_kernel = []
    channel_input =[]
    for i in range(0,channels):
        channel_kernel.append(kernel)
        channel_input.append(input)

        


    result_torch = pyTorch_conv(channel_input, channel_kernel, padding, stride)

    result_devito = conv_devito_2D_channels(channel_input, channel_kernel, padding, stride)

    print ("torch",result_torch[0])

    print ("devito",result_devito)


    print("Do they match", np.allclose(result_devito, result_torch))

    #assert(np.allclose(result_devito, result_torch))
#test_Devito_pyTorch(5,custom_kernel,4,2)

test_Devito_pyTorch_channels(5,custom_kernel,3,4,2)
