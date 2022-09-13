from devito import Function, sum, Eq,Operator,SpaceDimension, sumall
import numpy as np

data = np.random.randn(1,1,5,5)
input_dimensions = [SpaceDimension("d_"+str(x)) for x in range(0,4)]
input_func = Function(name=("Input_F"), shape=data.shape, dimensions=input_dimensions)
res_func = Function(name=("Result_F"), shape=(1,),dimensions=(SpaceDimension("res_d"),))
eqs = [Eq(res_func,  sumall(input_func))]
input_func.data[:] = data
op = Operator(eqs)
op.apply()
print(res_func.data) 

    
