import devito
from devito import Grid, Function, TimeFunction, Constant, Eq, Inc, \
    ConditionalDimension, SpaceDimension, IncrDimension, Operator
from matplotlib.pyplot import grid
import numpy as np
from sympy import Sum, factor, summation

import sympy as sp
from itertools import product



'''
    Input should be a 3D matrix like 3x512x512 image where 3 is the channels and 512x512 is the size of image.
    Similary kernel should be be 3D matrix where 3x5x5, 3 is the channel which 
    should always match the channels in input image, 5x5 is the kernel filter.
    Assumptioms - kernel would be of square size like 5x5, 3x3, same goes for image
'''

def conv_devito_2D(input,kernel, padding,stride, print_code = True):

    no_of_kernels = len(kernel)
    channels = len(kernel[0])
    input_size = len(input[0][0])
    kernel_size = len(kernel[0][0])
    result_size = (input_size-kernel_size + 2 *padding)//stride + 1 

    # input data function
    I = Function(name="Input_F",shape=(channels,input_size,input_size), dimensions =(SpaceDimension("input_d1"),SpaceDimension("input_d2"),SpaceDimension("input_d3")), space_order= padding, dtype=np.float64)

    # function for kernel
    K = Function(name="Kernel_F", shape=(no_of_kernels,channels,kernel_size,kernel_size), dimensions =(SpaceDimension("kernel_d1"),SpaceDimension("kernel_d2"),SpaceDimension("kernel_d3"), SpaceDimension("kernel_d4")), space_order=0, dtype=np.float64)

    # Result for convolution
    R = Function(name="Result_F", shape=(no_of_kernels, result_size,result_size), dimensions =(SpaceDimension("r_d1"),SpaceDimension("r_d2"),SpaceDimension("r_d3")), space_order=0, dtype=np.float64)

    #i for input dimension, k for kernel dimension, r for result dimension (2D array)
    r3, r1, r2 = R.dimensions
    k4,k3, k1, k2 = K.dimensions

    I.data [:] = input
    K.data [:]= kernel
    R.data [:]= 0

    off_sets = [x  for x in range(0,kernel_size )]
    off_sets_channels = [x  for x in range(0,channels )]

    k1_offsets = k2_offsets = [x for x in off_sets]
    k3_offsets = [x for x in off_sets_channels]
    #no_kernels_offsets = [k4+x for x in range(0,no_of_kernels)]


    k_indices = product(k3_offsets,k1_offsets,k2_offsets)

    w = sp.Matrix([K[(k4,*x)] for x in k_indices]) 

    r1_offsets = [(r1*stride+x)-padding for x in off_sets]
    r2_offsets = [r2*stride+x -padding for x in off_sets]
    i3_offsets = [x for x in off_sets_channels]


    r_indicies = product(i3_offsets,r1_offsets,r2_offsets)
    r = sp.Matrix([I[x] for x in r_indicies])

    sten = w.dot(r)

    eq = Eq(R[r3,r1,r2], sten)

    op = Operator(eq, opt="noop")
    
    op.apply()

    if print_code:
        print(op.ccode)

    return R.data