import devito
from devito import Grid, Function, TimeFunction, Constant, Eq, Inc, \
    ConditionalDimension, SpaceDimension, IncrDimension, Operator
from matplotlib.pyplot import grid
import numpy as np
from sympy import Sum, factor, summation

import sympy as sp
from itertools import product

def conv_devito_2D (input,kernel,padding,stride, print_code = True):

    input_size = len(input)
    kernel_size = len(kernel)
    result_size = (input_size-kernel_size + 2 *padding)//stride + 1 

    # input data function
    I = Function(name="Input_F",shape=(input_size,input_size), dimensions =(SpaceDimension("input_d1"),SpaceDimension("input_d2")), space_order= padding, dtype=np.float64)

    # function for kernel
    K = Function(name="Kernel_F", shape=(kernel_size,kernel_size), dimensions =(SpaceDimension("kernel_d1"),SpaceDimension("kernel_d2")), space_order=0, dtype=np.float64)

    # Result for convolution
    R = Function(name="Result_F", shape=(result_size,result_size), dimensions =(SpaceDimension("r_d1"),SpaceDimension("r_d2")), space_order=0, dtype=np.float64)

    #i for input dimension, k for kernel dimension, r for result dimension (2D array)
    r1, r2 = R.dimensions

    I.data [:] = input
    K.data [:]= kernel
    R.data [:]= 0


    off_sets = [x  for x in range(0,kernel_size )]

    k1_offsets = k2_offsets = [x for x in off_sets]

    k_indicies = product(k1_offsets,k2_offsets)

    w = sp.Matrix([K[x] for x in k_indicies]) 

    r1_offsets = [(r1*stride+x)-padding for x in off_sets]
    r2_offsets = [r2*stride+x -padding for x in off_sets]

    r_indicies = product(r1_offsets,r2_offsets)

    r = sp.Matrix([I[x] for x in r_indicies])

    sten = w.dot(r)

    eq = Eq(R[r1,r2], sten)

    op = Operator(eq, opt="noop")
    op.apply()

    if print_code:
        print(op.ccode)

    return R.data


def conv_devito_2D_channels (input,kernel, padding,stride, print_code = True):


    channels = len(kernel)
    input_size = len(input[0])
    kernel_size = len(kernel[0])
    result_size = (input_size-kernel_size + 2 *padding)//stride + 1 

    # input data function
    I = Function(name="Input_F",shape=(channels,input_size,input_size), dimensions =(SpaceDimension("input_d1"),SpaceDimension("input_d2"),SpaceDimension("input_d3")), space_order= padding, dtype=np.float64)

    # function for kernel
    K = Function(name="Kernel_F", shape=(channels,kernel_size,kernel_size), dimensions =(SpaceDimension("kernel_d1"),SpaceDimension("kernel_d2"),SpaceDimension("kernel_d3")), space_order=0, dtype=np.float64)

    # Result for convolution
    R = Function(name="Result_F", shape=(result_size,result_size), dimensions =(SpaceDimension("r_d1"),SpaceDimension("r_d2")), space_order=0, dtype=np.float64)

    #i for input dimension, k for kernel dimension, r for result dimension (2D array)
    r1, r2 = R.dimensions
    I.data [:] = input
    K.data [:]= kernel
    R.data [:]= 10

    off_sets = [x  for x in range(0,kernel_size )]
    off_sets_channels = [x  for x in range(0,channels )]

    k1_offsets = k2_offsets = [x for x in off_sets]
    k3_offsets = [x for x in off_sets_channels]

    k_indices = product(k3_offsets,k1_offsets,k2_offsets)

    w = sp.Matrix([K[x] for x in k_indices]) 

    r1_offsets = [(r1*stride+x)-padding for x in off_sets]
    r2_offsets = [r2*stride+x -padding for x in off_sets]
    i3_offsets = [x for x in off_sets_channels]

    r_indicies = product(i3_offsets,r1_offsets,r2_offsets)


    r = sp.Matrix([I[x] for x in r_indicies])

    sten = w.dot(r)

    eq = Eq(R[r1,r2], sten)

    op = Operator(eq, opt="noop")
    
    op.apply()

    if print_code:
        print(op.ccode)

    return R.data