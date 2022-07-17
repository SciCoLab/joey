import devito
from devito import Grid, Function, Constant, Eq, Inc, \
    ConditionalDimension, SpaceDimension, Operator
from matplotlib.pyplot import grid
import numpy as np
from sympy import Sum

import sympy as sy

input  = []
c=0;
#generate input data = [[0..5],[6..10],...[20..25]]
for i in range(0,5):
    temp =[]
    for j in range(0,5):
        temp.append(c)
        c = c+1;
    input.append(temp)

custom_kernel = [[0,0,0] ,[1,0,-1],[0,0,0]]


custom_input_T= input;


gridI = Grid(shape=(5,5), dimensions =(SpaceDimension("input_d1"),SpaceDimension("input_d2")))

gridK = Grid(shape=(3,3), dimensions =(SpaceDimension("kernel_d1"),SpaceDimension("kernel_d2")))

gridR = Grid(shape=(3,3), dimensions =(SpaceDimension("r_d1"),SpaceDimension("r_d2")))


# input data function
I = Function(name="Input_F", grid=gridI, space_order=0,dtype=np.float64)

# function for kernel
K = Function(name="Kernel_F", grid=gridK, space_order=0,dtype=np.float64)

# Result for convolution
R = Function(name="Result_F", grid=gridR, space_order=0,dtype=np.float64)

#i for input dimension, k for kernel dimension, r for result dimension (2D array)
i1, i2 = I.dimensions
r1, r2 = R.dimensions
k1, k2 = K.dimensions

I.data [:] = input

K.data [:]= custom_kernel

R.data [:]= 0

# single point in result R(x.y) =SUM(K[0..k1,0..k2] * I[ r1  , r2 + k2])
conv = K[k1,k2] * I[ r1  , r2 + k2]


# using sympy Sum operator to sum the kernel * input matrix
i = sy.symbols('i')

eqs = Sum(conv, (i, 0, 3))

eq = Eq(R[r1,r2],eqs)
op = Operator(eqs, opt="noop")

print(op.ccode)


print(R.data)


