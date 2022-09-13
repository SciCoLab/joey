from devito import Function, sum, Eq,Operator,SpaceDimension, sumall
import devito as dv
import numpy as np

def sumtt(f, dims=None):
    """
    Compute the sum of the Function data over specified dimensions.
    Defaults to sum over all dmensions

    Parameters
    ----------
    f : Function
        Input Function.
    dims : Dimension or tuple
        Dimensions to sum over
    """
    if dims is None or dims == f.dimensions:
        return sumall(f)

    dims = dv.tools.as_tuple(dims)
    new_dims = tuple(d for d in f.dimensions if d not in dims)
    shape = tuple(f._size_domain[d] for d in new_dims)
  
    out = dv.Function(name="%ssum" % f.name, grid=f.grid,
                          space_order=f.space_order, shape=shape,
                          dimensions=new_dims)
    kw = {}
    if f.is_TimeFunction:
        kw = {'time_m': 0, 'time_M': f.time_order}
    op = dv.Operator(dv.Eq(out, out + f))
    return op

data = np.random.randn(1,1,5,5)
input_dimensions = [SpaceDimension("d_"+str(x)) for x in range(0,4)]
input_func = Function(name=("Input_F"), shape=data.shape, dimensions=input_dimensions)
res_func = Function(name=("Result_F"), shape=(1,),dimensions=(SpaceDimension("res_d"),))
eqs = [Eq(res_func, res_func+input_func)]
input_func.data[:] = data
op = Operator(eqs)
op.apply()
print(res_func.data) 
print(np.sum(data)) 

    
