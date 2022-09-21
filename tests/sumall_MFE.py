from devito import Function,Operator,SpaceDimension
import devito as dv
import numpy as np
from devito import configuration
configuration['compiler'] = 'nvc++'
configuration['language'] = 'openacc'
configuration['log-level'] = 'DEBUG'
configuration['platform'] = 'nvidiaX'
data = np.random.randn(4000,4000) ; k = np.random.randn(5,5)
input_dimensions = [SpaceDimension("inp_I_"+str(x)) for x in range(0,2)]
kernel_dimensions = [SpaceDimension("inp_K_"+str(x)) for x in range(0,2)]
result_dimensions = [SpaceDimension("inp_R_"+str(x)) for x in range(0,2)]
input_func = Function(name=("Input_F"), shape=data.shape, dimensions=input_dimensions)
kernel_func = Function(name=("Kernel_F"), shape=k.shape, dimensions=kernel_dimensions)
res_func = Function(name=("Result_F"), shape=(46,46),dimensions=result_dimensions)
input_func.data[:] = data; kernel_func.data[:]=k
rhs = kernel_func[kernel_dimensions] * input_func[result_dimensions[0] + kernel_dimensions[0],result_dimensions[1] + kernel_dimensions[1]]
eqs = [dv.Inc(res_func, rhs)]
op = Operator(eqs)
print(op.ccode)
op.apply(deviceid=0)
print(res_func.data)
    
