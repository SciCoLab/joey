import devito
from devito import Grid, Function, TimeFunction, Constant, Eq, Inc, \
    ConditionalDimension, SpaceDimension, IncrDimension, Operator
from matplotlib.pyplot import grid
import numpy as np
from sympy import Sum, factor, summation

import sympy as sp
from itertools import product

def conv_devito_2D (input,kernel,stride,padding, print_code = True):

    input_size = len(input)

    kernel_size = len(kernel)


    result_size = (input_size-kernel_size + 2 *padding)//stride + 1 


    i_1 = SpaceDimension("input_d1")

    i_2 = SpaceDimension("input_d2")

    gridI = Grid(shape=(input_size,input_size), dimensions =(i_1,i_2))

    gridK = Grid(shape=(kernel_size,kernel_size), dimensions =(SpaceDimension("kernel_d1"),SpaceDimension("kernel_d2")))


    gridR = Grid(shape=(result_size,result_size), dimensions =(SpaceDimension("r_d1"),SpaceDimension("r_d2")))


    # input data function
    I = Function(name="Input_F", grid=gridI, space_order= padding, dtype=np.float64)

    # function for kernel
    K = Function(name="Kernel_F", grid=gridK, space_order=0, dtype=np.float64)

    # Result for convolution
    R = Function(name="Result_F", grid=gridR, space_order=0, dtype=np.float64)

    #i for input dimension, k for kernel dimension, r for result dimension (2D array)
    i1, i2 = I.dimensions
    r1, r2 = R.dimensions
    k1, k2 = K.dimensions


    I.data [:] = input

    K.data [:]= kernel

    R.data [:]= 0

    off_sets = [-1,0,1]
    if padding ==0:
        off_sets = [0,1,2]
    off_sets = [0,1,2]

    k1_offsets = [k1+x for x in off_sets]

    k2_offsets = [k2+x for x in off_sets]

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


# def conv_devito_2D_incr (input,kernel,stride,padding, print_code = True):

#     input_size = len(input)

#     kernel_size = len(kernel)


#     result_size = (input_size-kernel_size + 2 *padding)//stride + 1 


  
#     i_1 = SpaceDimension("input_d1")

#     i_2 = SpaceDimension("input_d2")

#     inr_1 = IncrDimension("in_1", parent=i_1, _min =0, _max = input_size, step = stride)

#     inr_2 = IncrDimension("in_2", parent=i_2, _min =0, _max = input_size, step = stride)

#     gridI = Grid(shape=(input_size,input_size), dimensions =(i_1,i_2))

#     gridK = Grid(shape=(kernel_size,kernel_size), dimensions =(SpaceDimension("kernel_d1"),SpaceDimension("kernel_d2")))

   
#     gridR = Grid(shape=(result_size,result_size), dimensions =(SpaceDimension("r_d1"),SpaceDimension("r_d2")))
#     gridR = Grid(shape=(result_size,result_size), dimensions =(ConditionalDimension("r_d1",parent=inr_1, factor = stride), ConditionalDimension("r_d2",parent=inr_2, factor = stride)))



#     # input data function
#     I = Function(name="Input_F", grid=gridI, shape=(input_size,input_size), dimensions = (inr_1, inr_2), space_order= padding, dtype=np.float64)

#     # function for kernel
#     K = Function(name="Kernel_F", grid=gridK, space_order=0, dtype=np.float64)

#     # Result for convolution
#     R = Function(name="Result_F", grid=gridR, space_order=0, dtype=np.float64)

#     #i for input dimension, k for kernel dimension, r for result dimension (2D array)
#     i1, i2 = I.dimensions
#     r1, r2 = R.dimensions
#     k1, k2 = K.dimensions


#     I.data [:] = input

#     K.data [:]= kernel

#     R.data [:]= 0

#     off_sets = [0,1,2]

#     k1_offsets = [k1+x for x in off_sets]

#     k2_offsets = [k2+x for x in off_sets]

#     k_indicies = product(k1_offsets,k2_offsets)



#     w = sp.Matrix([K[x] for x in k_indicies]) 

#     r1_offsets = [i1+x-padding for x in off_sets]

#     r2_offsets = [i2+x-padding for x in off_sets]

#     r_indicies = product(r1_offsets,r2_offsets)

#     r = sp.Matrix([I[x] for x in r_indicies])


#     sten = w.dot(r)


#     eq = Eq(R[r1,r2], sten)

#     op = Operator(eq, opt="noop")

#     op.apply()

#     if print_code:
#         print(op.ccode)


#     return R.data   



# def conv_devito_2D_defaultStride(input,kernel,stride,padding, print_code = True):

#     input_size = len(input)

#     kernel_size = len(kernel)


#     result_size = (input_size-kernel_size + 2 *padding)//stride + 1 


#     i_1 = SpaceDimension("input_d1")
    
#     i_2 = SpaceDimension("input_d2")

#     gridI = Grid(shape=(input_size,input_size), dimensions =(i_1,i_2))

#     gridK = Grid(shape=(kernel_size,kernel_size), dimensions =(SpaceDimension("kernel_d1"),SpaceDimension("kernel_d2")))

#     gridR = Grid(shape=(result_size,result_size), dimensions =(SpaceDimension("r_d1"),SpaceDimension("r_d2")))



#     # input data function
#     I = Function(name="Input_F", grid=gridI, space_order= padding, dtype=np.float64)

#     # function for kernel
#     K = Function(name="Kernel_F", grid=gridK, space_order=0, dtype=np.float64)

#     # Result for convolution
#     R = Function(name="Result_F", grid=gridR, space_order=0, dtype=np.float64)

#     #i for input dimension, k for kernel dimension, r for result dimension (2D array)
#     i1, i2 = I.dimensions
#     r1, r2 = R.dimensions
#     k1, k2 = K.dimensions


#     I.data [:] = input

#     K.data [:]= kernel

#     R.data [:]= 0

#     off_sets = [0,1,2]

#     k1_offsets = [k1+x for x in off_sets]

#     k2_offsets = [k2+x for x in off_sets]

#     k_indicies = product(k1_offsets,k2_offsets)



#     w = sp.Matrix([K[x] for x in k_indicies]) 

#     r1_offsets = [r1+x-padding for x in off_sets]

#     r2_offsets = [r2+x-padding for x in off_sets]

#     r_indicies = product(r1_offsets,r2_offsets)

#     r = sp.Matrix([I[x] for x in r_indicies])


#     sten = w.dot(r)


#     eq = Eq(R[r1,r2], sten)

#     op = Operator(eq, opt="noop")

#     op.apply()

#     if print_code:
#         print(op.ccode)


#     return R.data   