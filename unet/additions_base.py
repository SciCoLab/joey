"""
Additional  Base functions
The code was written originally here, with its version history:
https://github.com/cha-tzi/devito/blob/ml/devito/ml/base.py
###############################################################
These functions are necessary for solving the padding issue. 
Padding is only applied to Spatial dimensions. 
However, there is no function to call that automatically
creates multiple dimensions, the same way dimensions(names) works. 
Therefore we need to call the space_dim_allocator for each spatial dimension. 
The need for the single_dim_allocator function arises because the dimensions() 
function outputs a tuple, while the SpaceDimention() function outputs just 
the dimension, so it is not possible to combine them.
"""
from abc import ABC, abstractmethod
from devito import Operator, Function, dimensions, Dimension, SpaceDimension
from numpy import array

index = 0
dim_index = 0

def default_name_allocator():
    global index
    name = 'f' + str(index)
    index += 1
    return name

def default_dim_allocator(count):
    global dim_index
    names = ''
    for i in range(count):
        names += 'd' + str(dim_index) + ' '
        dim_index += 1
    names = names[:-1]
    return dimensions(names)

def single_dim_allocator():
    global dim_index
    names = ''
    names += 'd' + str(dim_index) + ' '
    dim_index += 1
    names = names[:-1]
    return Dimension(names)

def space_dim_allocator():
    global dim_index
    names = ''
    names += 'd' + str(dim_index) + ' '
    dim_index += 1
    names = names[:-1]
    return SpaceDimension(names)