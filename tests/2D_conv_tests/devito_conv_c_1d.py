import devito
from devito import Grid, Function, Constant, Eq, Inc, \
    ConditionalDimension, SpaceDimension, Operator
from matplotlib.pyplot import grid
import numpy as np
from sympy import Sum, summation

import sympy as sp

input  = []
c=0;
#generate input data = [[0..5],[6..10],...[20..25]]
for i in range(1,51):
    input.append(i)

custom_kernel = [1.4,1.5,1.4]


custom_input_T= input;


gridI = Grid(shape=(50,),dimensions =(SpaceDimension("input_d1"),))

gridK = Grid(shape=(3,),dimensions =(SpaceDimension("kernel_d1"),))

gridR = Grid(shape=(48,),dimensions =(SpaceDimension("result_d1"),))


# input data function
I = Function(name="Input_F", grid=gridI, space_order=0,dtype=np.float64)

# function for kernel
K = Function(name="Kernel_F", grid=gridK, space_order=0,dtype=np.float64)

# Result for convolution
R = Function(name="Result_F", grid=gridR, space_order=0,dtype=np.float64)

#i for input dimension, k for kernel dimension, r for result dimension (2D array)
i1 = I.dimensions[0]
k1 = K.dimensions[0]
r1 = R.dimensions[0]

I.data [:] = input

K.data [:]= custom_kernel

R.data [:]= 0

# eq = Eq(R[r1],custom_kernel[0]*I[r1-1]+custom_kernel[1]*I[r1]+ custom_kernel[2] *I[r1+1])


# op = Operator(eq, opt="noop")

# print(op.ccode)


# print(R.data)



w = sp.Matrix([K[k1-1], K[k1], K[k1+1]])
p = sp.Matrix([I[r1-1], I[r1], I[r1+1]])
sten = w.dot(p)
eq = Eq(R[r1], sten)

op = Operator(eq, opt="noop")

print(op.ccode)

op.apply()

print("data norm",np.linalg.norm(I.data))


print("resul norm",np.linalg.norm(R.data))
