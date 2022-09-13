from itertools import product
from tkinter import Y
from devito import Max, Function, sum, Eq,Max,SubDimension,Operator, Constant, Inc, ConditionalDimension, SpaceDimension, sumall
from sympy import And
import numpy as np
import sympy as sp
from joey import Layer
from joey import get_name
from abc import abstractmethod

from joey import default_name_allocator as alloc
from joey import default_dim_allocator as dim_alloc


class ConvV2(Layer):
    """
    A Layer subclass corresponding to convolution layer (mathematically,
    it performs a cross-correlation operation).

    Parameters
    ----------
    kernel_size : (int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    dimensions: int
        The dimension of conv operation i.e.
        if the convolution is 1d, 2d or so on
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size, dimensions,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        # Internal kernel size (self._kernel_size) is expressed as
        # (output channels / kernel count, input channels, rows, columns).
        self._dims = dimensions

        if (type(padding) is int):
            padding = tuple([padding] * self._dims)
        if (type(stride) is int):
            stride = tuple([stride] * self._dims)
        self._error_check(kernel_size, input_size, stride, padding,
                          strict_stride_check)

        self._kernel_size = (kernel_size[0], input_size[1], *kernel_size[1:])
        self._stride = stride
        self._padding = padding

        super().__init__(self._kernel_size, input_size, activation,
                         alloc, get_name,
                         generate_code)

    def _error_check(self, kernel_size, input_size, stride, padding,
                     strict_stride_check):
        if input_size is None or (len(input_size) != self._dims+2):
            raise Exception("Input size is incorrect")

        if kernel_size is None or (len(kernel_size) != self._dims+1):
            raise Exception("Kernel size is incorrect")

        if stride is None or (len(stride) != self._dims):
            raise Exception("Stride is incorrect")

        if padding is None or (len(padding) != self._dims):
            raise Exception("Padding is incorrect")

        for i in range(0, self._dims):

            if stride[i] < 1:
                raise Exception("Stride cannot be less than 1")

            if padding[i] < 0:
                raise Exception("Padding cannot be negative")

        if strict_stride_check:
            input_d = input_size[-self._dims+i] + 2 * padding[i]
            if (input_d - kernel_size[-self._dims+i]) % stride[i] != 0:
                raise Exception("Stride " + str(stride) + " is not "
                                "compatible with feature map, kernel and "
                                "padding sizes. If you want to proceed "
                                "anyway, set strict_stride_check=False "
                                "when instantiating this object")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  get_name):

        no_of_kernels = kernel_size[0]
        self.dim_dict = dim_dict = {3: 'depth', 2: 'height', 1: 'width'}

        dimensions = ['dbatch', 'dchannel']
        result_shape = []
        # generating  in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on
        for i in range(0, self._dims):
            result_d = (input_size[(-self._dims+i)] -
                        kernel_size[(-self._dims+i)] +
                        2 * self._padding[i])//self._stride[i] + 1
            result_shape.append(result_d)
            dimensions.append('d_'+dim_dict.get(self._dims-i, self._dims-i))

        result_shape = (input_size[0], no_of_kernels, *result_shape)

        # input data function
        input_dimensions = [SpaceDimension(
            get_name("Input_"+x)) for x in dimensions]

        input_func = Function(name=get_name("Input_F"), shape=(input_size),
                              dimensions=input_dimensions, space_order=(
            self._padding[0]), dtype=np.float64)

        # function for kernel
        kernel_dims = [SpaceDimension(get_name("kernel_"+x))
                       for x in dimensions]

        kernel_func = Function(name=get_name("Kernel_F"), shape=(kernel_size),
                               dimensions=(kernel_dims), space_order=0,
                               dtype=np.float64)

        # Result for convolution
        result_dimensions = [SpaceDimension(
            get_name("Result_"+x)) for x in dimensions]

        result_func = Function(name=get_name("Result_F"), shape=result_shape,
                               dimensions=result_dimensions, space_order=0,
                               dtype=np.float64)

        bias_dimensions = [SpaceDimension(get_name("bias_"+x)) for x in ['d']]

        bias = Function(name=get_name("bias_F"), shape=(
            kernel_size[0],), dimensions=bias_dimensions, space_order=0,
            dtype=np.float64)

        kernel_grad_dimensions = [SpaceDimension(
            get_name("kernel_grad"+x)) for x in dimensions]

        kernel_grad = Function(name="kgrad_%s" % name_allocator_func(), shape=(
            kernel_size), dimensions=kernel_grad_dimensions, space_order=0,
            dtype=np.float64)

        output_grad_dimensions = [SpaceDimension(
            get_name("output_grad"+x)) for x in dimensions]

        output_grad = Function(name="outgrad_%s" % name_allocator_func(
        ), shape=result_shape, dimensions=output_grad_dimensions,
            space_order=(self._kernel_size[-1]+self._padding[0]-1), dtype=np.float64)

        bias_grad_dimensions = [SpaceDimension(
            get_name("bias_"+x)) for x in ['d']]

        bias_grad = Function(name="bgrad_%s" % name_allocator_func(), shape=(
            kernel_size[0],), dimensions=bias_grad_dimensions, space_order=0,
            dtype=np.float64)

        return (kernel_func, input_func, result_func, bias, kernel_grad,
                output_grad, bias_grad)

    def execute(self, input_data, bias, kernel_data=None) -> np.array:
        if kernel_data is not None:
            self._K.data[:] = kernel_data

        self._bias.data[:] = bias
        self._I.data[:] = input_data
        self._R.data[:] = 0

        return super().execute()

    def equations(self):

        result_dimensions = self._R.dimensions
        bias = self._bias.dimensions
        k_dims_offsets = []
        for i in range(0, self._dims):
            k_dims_offsets.append(
                list(range(0, self._kernel_size[-self._dims + i])))

        off_sets_channels = list(range(0, self._kernel_size[1]))

        # indices of kernel matrix for convolution
        k_indices = product(off_sets_channels, * k_dims_offsets)

        r_dims_offsets = []

        # generating offsets in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on
        for i in range(0, self._dims):
            r_dim_offsets = [result_dimensions[-self._dims + i]
                             * self._stride[i]+x
                             - self._padding[i] for x in k_dims_offsets[i]]
            r_dims_offsets.append(r_dim_offsets)

        # indices of input based on resullt matrix for convolution
        r_indicies = product(off_sets_channels, *r_dims_offsets)

        weight_matrix = sp.Matrix(
            [self._K[(result_dimensions[1], *x)] for x in k_indices])

        r_indices_matrix = sp.Matrix(
            [self._I[(result_dimensions[0], *x)] for x in r_indicies])

        # stencil operation corresponding to the convolution
        sten = weight_matrix.dot(r_indices_matrix)

        eqs = [Eq(self._R[result_dimensions], sten)]
        eqs.append(Inc(self._R[result_dimensions], self._bias[bias]))
        if self._activation is not None:
            eqs.append(Eq(self._R[result_dimensions],
                          self._activation(self._R[result_dimensions])))
        return (eqs, [])

    def backprop_equations(self, prev_layer, next_layer):
        layer = self

        kernel_dims = layer.kernel_gradients.dimensions
        kernel_shape = layer.kernel_gradients.shape
        bias_dims = layer.bias_gradients.dimensions
        result_grad_dims = layer.result_gradients.dimensions
        result_grad_shape = layer.result_gradients.shape

        k_dims_offsets = []
        for i in range(0, self._dims):
            k_dims_offsets.append(
                list(range(result_grad_shape[-self._dims + i]-1, -1, -1)))

        # indices of kernel matrix for convolution
        k_indices = product(* k_dims_offsets)

        r_dims_offsets = []

        # generating offsets in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on

        for i in range(0, self._dims):
            r_dim_offsets = [kernel_dims[-self._dims + i] + x *
                self._stride[i] - self._padding[i] for x in k_dims_offsets[i]]
            r_dims_offsets.append(r_dim_offsets)

        # indices of input based on resullt matrix for convolution
        r_indicies = product(*r_dims_offsets)

        weight_matrix = sp.Matrix(
            [layer.result_gradients[(*result_grad_dims[0:1], kernel_dims[0], *x)] for x in k_indices])

        r_indices_matrix = sp.Matrix(
            [self._I[(result_grad_dims[0], kernel_dims[1], *x)] for x in r_indicies])

        # stencil operation corresponding to the convolution
        sten = weight_matrix.dot(r_indices_matrix)
        eqs = [Inc(layer.bias_gradients[bias_dims[0]],
                   layer.result_gradients[result_grad_dims])]
        eqs += [Inc(layer.kernel_gradients[(kernel_dims)],  sten)]

        if next_layer is not None:
            next_layer_dims = next_layer.result_gradients.dimensions

            k_dims_offsets = []
            for i in range(0, self._dims):
                k_dims_offsets.append(
                    list(range(kernel_shape[-self._dims + i]-1, -1, -1)))
            off_sets_channels = list(range(0, self._kernel_size[1]))

            # indices of kernel matrix for convolution
            k_indices = product(* k_dims_offsets)

            k_dims_offsets = []
            for i in range(0, self._dims):
                k_dims_offsets.append(
                    list(range(0, kernel_shape[-self._dims + i], 1)))
            r_dims_offsets = []

            # generating offsets in the order depth, height, width ,
            # hence arr[-3], arr[-2] and so on

            for i in range(0, self._dims):
                r_dim_offsets = [next_layer_dims[-self._dims + i] + x - (self._kernel_size[-1]-self._padding[0]-1) for x in k_dims_offsets[i]]
                r_dims_offsets.append(r_dim_offsets)

            # indices of input based on resullt matrix for convolution
            r_indicies = product(*r_dims_offsets)

            weight_matrix = sp.Matrix(
                [self._K[(kernel_dims[0],kernel_dims[1], *x)] for x in k_indices])

            r_indices_matrix = sp.Matrix(
                [layer.result_gradients[(next_layer_dims[0],kernel_dims[0], *x)] for x in r_indicies])

            # stencil operation corresponding to the convolution
            sten = weight_matrix.dot(r_indices_matrix)

            eqs += [Inc(next_layer.result_gradients[(next_layer_dims[0],kernel_dims[1],*next_layer_dims[2:])], sten,implicit_dims=(kernel_dims[0],next_layer_dims[2]))]

            + next_layer.activation.backprop_eqs(next_layer)


        return (eqs, [])


class Conv3D(ConvV2):
    """
    A Layer subclass corresponding to a 3D convolution layer (mathematically,
    it performs a cross-correlation operation).

    Parameters
    ----------
    kernel_size : (int, int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,
                 stride=(1, 1, 1), padding=(0, 0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 3
        super().__init__(kernel_size, input_size, dimensions,
                         stride, padding, activation, generate_code,
                         strict_stride_check)


class Conv2DV2(ConvV2):
    """
    A Layer subclass corresponding to a 2D convolution layer (mathematically,
    it performs a cross-correlation operation).

    Parameters
    ----------
    kernel_size : (int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 2
        super().__init__(kernel_size, input_size, dimensions,
                         stride, padding, activation, generate_code,
                         strict_stride_check)




class Pooling(Layer):
    """
    A Layer abstract subclass corresponding to a generic pooling layer.
    When you create a subclass of Pooling, you have to implement
    the following methods: equations(), backprop_equations().

    Parameters
    ----------
    kernel_size : (int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. TSymbol, sympifydata will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,  dimensions,
                 name_allocator_func=alloc, dim_allocator_func=dim_alloc,
                 stride=(1, 1), padding=(0, 0), activation=None,
                 generate_code=False, strict_stride_check=True):
        # Kernel size is expressed as (rows, columns).
        # Input size is expressed as (batch size, channels, rows, columns).
        self._dims = dimensions
        self._dim_dict = {3: 'depth', 2: 'height', 1: 'width'}

        if (type(padding) is int):
            padding = tuple([padding] * self._dims)
        if (type(stride) is int):
            stride = tuple([stride] * self._dims)
        self._error_check(kernel_size, input_size, stride, padding,
                          strict_stride_check)

        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        super().__init__(self._kernel_size, input_size, activation,
                         alloc, dim_alloc,
                         generate_code)

    def _set_padding_result_values(self, input_func, result_func, value=0):
        devito_func_dims = input_func.dimensions[2:]
        eqs = []
        for i, dim in enumerate(devito_func_dims):
            if self._padding[i] > 0:
                dim_l = SubDimension.left(name='sub%s_l' % dim.name, parent=dim,
                                      thickness=self._padding[i])
                eqs.append(Inc(input_func.subs({dim: dim_l}), value))
                dim_r = SubDimension.right(name='sub%s_r' % dim.name, parent=dim,
                                       thickness=self._padding[i])
                eqs.append(Inc(input_func.subs({dim: dim_r}), value))
        eqs.append(Eq(result_func, value))
        op = Operator(eqs)
        op.apply()
        return

    def _error_check(self, kernel_size, input_size, stride, padding,
                     strict_stride_check):
        if input_size is None or (len(input_size) != self._dims+2):
            raise Exception("Input size is incorrect")

        if kernel_size is None or (len(kernel_size) != self._dims):
            raise Exception("Kernel size is incorrect")

        if stride is None or (len(stride) != self._dims):
            raise Exception("Stride is incorrect")

        if padding is None or (len(padding) != self._dims):
            raise Exception("Padding is incorrect")

        for i in range(0, self._dims):

            if stride[i] < 1:
                raise Exception("Stride cannot be less than 1")

            if padding[i] < 0:
                raise Exception("Padding cannot be negative")

        if strict_stride_check:
            input_d = input_size[-self._dims+i] + 2 * padding[i]
            if (input_d - kernel_size[-self._dims+i]) % stride[i] != 0:
                raise Exception("Stride " + str(stride) + " is not "
                                "compatible with feature map, kernel and "
                                "padding sizes. If you want to proceed "
                                "anyway, set strict_stride_check=False "
                                "when instantiating this object")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):

        dimensions = ['dbatch', 'dchannel']
        result_shape = []
        input_size = list(input_size)
        # generating  in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on
        for i in range(0, self._dims):
            result_d = (input_size[(-self._dims+i)] -
                        kernel_size[(-self._dims+i)] +
                        2 * self._padding[i])//self._stride[i] + 1
            result_shape.append(result_d)
            input_size[(-self._dims+i)] += 2 * self._padding[i]
            dimensions.append(
                'd_'+self._dim_dict.get(self._dims-i, self._dims-i))

        result_shape = (*input_size[0:2], *result_shape)
        # input data function
        input_dimensions = [SpaceDimension("Input_"+x) for x in dimensions]

        input_func = Function(name="Input_F", shape=(input_size),
                              dimensions=input_dimensions,
                              space_order=0, dtype=np.float64)
        ind_dimensions = [SpaceDimension(get_name("Indices_"+x)) for x in dimensions]

        self._indices = Function(name='indices_maxpool', space_order=0, shape=(result_shape), dimensions=ind_dimensions,dtype=np.int16)

        # Result for pool layer
        result_dimensions = [SpaceDimension("Result_"+x) for x in dimensions]

        result_func = Function(name="Result_F", shape=result_shape,
                               dimensions=result_dimensions, space_order=0,
                               dtype=np.float64)

        output_grad_dimensions = [SpaceDimension(
            "output_grad"+x) for x in dimensions]

        output_grad = Function(name="outgrad_%s" % name_allocator_func(
        ), shape=result_shape, dimensions=output_grad_dimensions,
            space_order=0, dtype=np.float64)

        self._set_padding_result_values(
            input_func, result_func, value=np.finfo(np.float).min)

        return (None, input_func, result_func, None, None, output_grad, None)

    @property
    def stride(self):
        """Stride of the layer."""
        return self._stride

    @property
    def kernel_size(self):
        """The kernel size of the layer."""
        return self._kernel_size

    def execute(self, input_data):
        '''execute implementation'''
        dims = list(self._I.dimensions)
        indices = [slice(0, self._I.shape[0], 1),
                   slice(0, self._I.shape[1], 1)]
        [indices.append(slice(self._padding[i],
                              self._I.data.shape[2+i]-self._padding[i], 1))
            for i in range(self._dims)]

        self._I.data[tuple(indices)] = input_data

        return super().execute()

    @abstractmethod
    def equations(self):
        pass

    @abstractmethod
    def backprop_equations(self, prev_layer, next_layer):
        pass


class MaxPoolingV2(Pooling):
    """
    A Layer/Pooling subclass corresponding to a max pooling layer

    Parameters
    ----------
    See Pooling.__doc__.
    """

    def __init__(self, kernel_size, input_size, dims,
                 stride, padding, activation, generate_code,
                 strict_stride_check):
        self._indices = None
        self._forward_tmp_constants = None
        self._backward_tmp_constants = None
        super().__init__(kernel_size, input_size, dims, alloc, dim_alloc,
                         stride, padding, activation, generate_code,
                         strict_stride_check)

    def equations(self):
        result_dimensions = self._R.dimensions
        self._R.data[:] = np.finfo('float').min
        input_dims = [result_dimensions[0], result_dimensions[1]]
        args = []
        new_dims= []
        for i in range(0, self._dims):
            new_dim = SpaceDimension(
                name='d_kernel'+self._dim_dict.get(self._dims-i, self._dims-i))
            new_dims.append(new_dim)
            input_dims.append(
                result_dimensions[-self._dims + i]*self._stride[i]+new_dim)
            args.append((new_dim.name + '_M', self._kernel_size[i] - 1))
        ci = ConditionalDimension(name='ci', parent=new_dims[-1],
                          condition=And(self._I[input_dims]>self._R[result_dimensions]))
        index = 0
        exp_dims =[]
        for i in range (0,self._dims-1):
            prod = np.prod(self._I.shape[3:]) if i==0 else np.prod(self._I.shape[3:-i])
            index +=  prod * input_dims[-self._dims+i] 
            exp_dims.append(new_dims[-self._dims+i])
        exp_dims.append(ci)
        eqs = [Eq(self._indices[result_dimensions], index+(input_dims[-1]) ,implicit_dims=(exp_dims))]

        eqs += [Eq(self._R[result_dimensions], Max(
            self._R[result_dimensions], self._I[input_dims], evaluate=False))]

        # eqs += [Eq(self._indices[input_dims], 0, implicit_dims=(ci,))]


        return (eqs, args)

    def backprop_equations(self, prev_layer, next_layer):
        if next_layer is None:
            return ([], [])
        res_dims = self.result_gradients.dimensions
        next_layer_dims = next_layer.result_gradients.dimensions
        z = self._indices[res_dims]
        indices_const = []
        for i in range (0,self._dims):
            indices_const.append(Constant(name=get_name("pool_backprop_c_"+str(i)), dtype=np.int32))
        input_shape = self._I.shape
        eqs = []
        prod = 1
        for i in range (0,self._dims):
            prod = np.prod(self._I.shape[3:]) if i==0 else np.prod(self._I.shape[3:-i])
            eqs += [Eq(indices_const[i], (z//prod)-self._padding[i])]
            z = z %prod   
       
        # eqs.append(Eq(indices_const[-1] ,(z%input_shape[-1])-self._padding[-1]))
        eqs += [Inc(next_layer.result_gradients[(*res_dims[:2],*indices_const)], self.result_gradients[res_dims])]
        return (eqs,[])



class MaxPooling2D(MaxPoolingV2):

    """
    A Layer subclass corresponding to a 2D Max pool layer,
    which calls the generic pooling class.
    Parameters
    ----------
    kernel_size : (int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 2
        super().__init__(kernel_size, input_size, dimensions,
                         stride, padding, activation, generate_code,
                         strict_stride_check)


class MaxPooling3D(MaxPoolingV2):

    """
    A Layer subclass corresponding to a 2D Max pool layer,
    which calls the generic pooling class.
    Parameters
    ----------
    kernel_size : (int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (depth, rows, columns).
    input_size : (int, int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, depth, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 3
        super().__init__(kernel_size, input_size, dimensions,
                         stride, padding, activation, generate_code,
                         strict_stride_check)

class InstanceNorm(Layer):
    """
    A Layer subclass corresponding to convolution layer (mathematically,
    it performs a cross-correlation operation).

    Parameters
    ----------
    kernel_size : (int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    dimensions: int
        The dimension of conv operation i.e.
        if the convolution is 1d, 2d or so on
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, input_size, dimensions,
                 activation, generate_code,
                 strict_stride_check):
        # Internal kernel size (self._kernel_size) is expressed as
        # (output channels / kernel count, input channels, rows, columns).
        self._dims = dimensions

        self._error_check(input_size)

        super().__init__(None, input_size, activation,
                         alloc, dim_alloc,
                         generate_code)

    def _error_check(self, input_size):
        if input_size is None or (len(input_size) != self._dims+2):
            raise Exception("Input size is incorrect")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):

        dim_dict = {3: 'depth', 2: 'height', 1: 'width'}

        dimensions = ['dbatch', 'dchannel']
        mean_dimensions = [SpaceDimension("Mean_"+x) for x in dimensions]

        self._mean = Function(name=get_name("mean_func"), shape=(input_size[0:2]),
                              dimensions=mean_dimensions, space_order=0, dtype=np.float64)
        var_dimensions = [SpaceDimension("Var_"+x) for x in dimensions]

        self._var = Function(name=get_name("var_func"), shape=(input_size[0:2]),
                             dimensions=var_dimensions, space_order=0, dtype=np.float64)
        result_shape = []
        # generating  in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on
        for i in range(0, self._dims):
            result_d = (input_size[(-self._dims+i)])
            result_shape.append(result_d)
            dimensions.append('d_'+dim_dict.get(self._dims-i, self._dims-i))

        result_shape = (input_size[0], input_size[1], *result_shape)

        # input data function
        input_dimensions = [SpaceDimension("Input_"+x) for x in dimensions]

        input_func = Function(name=get_name("Input_F"), shape=(input_size),
                              dimensions=input_dimensions, space_order=0, dtype=np.float64)

        # Result for convolution
        result_dimensions = [SpaceDimension("Result_"+x) for x in dimensions]

        result_func = Function(name=get_name("Result_F"), shape=result_shape,
                               dimensions=result_dimensions, space_order=0,
                               dtype=np.float64)

        bias_dimensions = [SpaceDimension("bias_"+x) for x in ['d']]

        bias = Function(name=get_name("bias_F"), shape=(
            input_size[1],), dimensions=bias_dimensions, space_order=0,
            dtype=np.float64)

        output_grad_dimensions = [SpaceDimension(
            "output_grad"+x) for x in dimensions]

        output_grad = Function(name="outgrad_%s" % name_allocator_func(
        ), shape=result_shape, dimensions=output_grad_dimensions,
            space_order=0, dtype=np.float64)

        bias_grad_dimensions = [SpaceDimension("bias_"+x) for x in ['d']]

        bias_grad = Function(name="bgrad_%s" % name_allocator_func(), shape=(
            input_size[1],), dimensions=bias_grad_dimensions, space_order=0,
            dtype=np.float64)

        return (None, input_func, result_func, bias, None,
                output_grad, bias_grad)

    def execute(self, input_data) -> np.array:

        self._I.data[:] = input_data
        self._R.data[:] = 0

        return super().execute()

    def equations(self):

        result_dimensions = self._R.dimensions
        result_shape = self._R.shape
        bias = self._bias.dimensions
        self.offsets = k_dims_offsets = []
        for i in range(0, self._dims):
            k_dims_offsets.append(
                list(range(0, result_shape[-self._dims + i])))

        # indices of kernel matrix for convolution
        k_indices = product(* k_dims_offsets)

        self.temp_ones = temp_func = Function(name="Ones_Filter", shape=result_shape,
                             dimensions=result_dimensions, space_order=0,
                             dtype=np.float64)
        temp_func_i = Function(name=get_name("temp_inp"), shape=result_shape,
                               dimensions=result_dimensions, space_order=0,
                               dtype=np.float64)

        # indices of input based on resullt matrix for convolution
        r_indicies = product(*k_dims_offsets)
        temp_func.data[:] = 1
        weight_matrix = sp.Matrix(
            [temp_func[(*result_dimensions[:2], *x)] for x in k_indices])

        r_indices_matrix = sp.Matrix(
            [self._I[(*result_dimensions[:2], *x)] for x in r_indicies])
        N = np.prod(result_shape[2:])
        # stencil operation corresponding to the convolution with kernel of input_shape with value to simulate sum of input_mat
        sum_input_sten = weight_matrix.dot(r_indices_matrix)

        mean = (sum_input_sten/N)

        '''
        .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}}

        '''

        # deviation from mean
        eqs = [Eq(self._mean[result_dimensions[0:2]], mean)]
        eqs += [Eq(self._R[result_dimensions],
                   self._I[result_dimensions] - mean)]
        r_indicies = product(*k_dims_offsets)
        r_indices_matrix = sp.Matrix(
            [self._R[(*result_dimensions[0:2], *x)]**2 for x in r_indicies])

        sum_var_stencil = r_indices_matrix.dot(weight_matrix)
        eqs += [Eq(self._var[result_dimensions[0:2]], sum_var_stencil/N)]

        eqs += [Eq(temp_func_i[result_dimensions], sum_var_stencil/N)]
        epsilon = 0.00001
        eqs += [Eq(self._R[result_dimensions], self._R[result_dimensions] /
                   sp.sqrt(temp_func_i[result_dimensions]+epsilon))]

        #eqs.append(Inc(self._R[result_dimensions], self._bias[bias]))

        if self._activation is not None:
            eqs.append(Eq(self._R[result_dimensions],
                          self._activation(self._R[result_dimensions])))
        return (eqs, [])

    def backprop_equations(self, prev_layer, next_layer):
        layer = self

        var = layer._var
        dims = layer.result_gradients.dimensions
        result_shape = layer.result_gradients.shape
        N = np.prod(result_shape[2:])
        if next_layer is not None:
            next_dims = next_layer.result_gradients.dimensions
            # TODO: Better names for these dimensions
            layer.temp_func = temp_func = Function(name="temp_func_layer1", shape=result_shape,
                             dimensions=dims, space_order=0,
                             dtype=np.float64)
            eqs = [Eq(temp_func[dims], layer.result_gradients[dims]*layer.result[dims])]

            e1 =  (N)*(layer.result_gradients[dims]) 
            
            temp_var = Function(name="var_layer1", shape=var.shape,
                             dimensions=var.dimensions, space_order=0,
                             dtype=np.float64)
            
            r_indicies = product(*self.offsets)
            k_indices = product(*self.offsets)
           
            weight_matrix = sp.Matrix(
            [self.temp_ones[(*dims[:2], *x)] for x in k_indices])
            r_indices_matrix = sp.Matrix(
            [temp_func[(*dims[:2], *x)] for x in r_indicies])
            
            eqs += [Eq(temp_var[dims[0:2]],  weight_matrix.dot(r_indices_matrix))]
            
            r_indicies = product(*self.offsets)
            r_indices_matrix = sp.Matrix(
            [layer.result_gradients[(*dims[:2], *x)] for x in r_indicies])

            e3 = layer.result[dims] * temp_var[dims[0:2]]

            inv_var = 1/sp.sqrt(layer._var[dims[0:2]])
            e4 = e1-(weight_matrix.dot(r_indices_matrix))-e3
            eqs += [Eq(next_layer.result_gradients[dims],  e4 * inv_var * (1/N))]
            
        return (eqs, [])


class InstanceNorm3D(InstanceNorm):
    """
    A Layer subclass corresponding to a 3D convolution layer (mathematically,
    it performs a cross-correlation operation).

    Parameters
    ----------
    kernel_size : (int, int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, input_size,
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 3
        super().__init__(input_size, dimensions,
                         activation, generate_code,
                         strict_stride_check)


class InstanceNorm2D(InstanceNorm):
    """
    A Layer subclass corresponding to a 2D convolution layer (mathematically,
    it performs a cross-correlation operation).

    Parameters
    ----------
    kernel_size : (int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, input_size,
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 2
        super().__init__(input_size, dimensions,
                         activation, generate_code,
                         strict_stride_check)


class UpSample(Layer):
    """
    A Layer subclass corresponding to upsampling layer
    Parameters
    ----------
    scale_factor : int or tuple of (input_dims)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (depth, rows, columns) or (rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    dimensions: int
        The dimension of conv operation i.e.
        if the convolution is 1d, 2d or so on
    """

    def __init__(self, scale_factor, input_size,
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        # Internal kernel size (self._kernel_size) is expressed as
        # (output channels / kernel count, input channels, rows, columns).

        self._dims = len(input_size)-2
        if type(scale_factor) is int:
            scale_factor = tuple([scale_factor]*self._dims)

        self._error_check(scale_factor, input_size)

        self._scale_factor = scale_factor
        super().__init__(None, input_size, activation,
                         alloc, dim_alloc,
                         generate_code)

    def _error_check(self, scale_factor, input_size):
        if input_size is None or (len(input_size) < 3):
            raise Exception("Input size is incorrect")

        if scale_factor is None or (len(scale_factor) != len(input_size)-2):
            raise Exception("Kernel size is incorrect")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):

        self._dim_dict = dim_dict = {3: 'depth', 2: 'height', 1: 'width'}
        self._dims = len(input_size)-2
        dimensions = ['dbatch', 'dchannel']
        result_shape = []
        # generating  in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on
        for i in range(0, self._dims):
            result_d = (input_size[(-self._dims+i)] * self._scale_factor[i])
            result_shape.append(result_d)
            dimensions.append('d_'+dim_dict.get(self._dims-i, self._dims-i))

        result_shape = (*input_size[0:2], *result_shape)

        # input data function
        input_dimensions = [SpaceDimension("Input_"+x) for x in dimensions]

        input_func = Function(name="Input_F",
                              shape=(input_size), space_order=0,
                              dimensions=input_dimensions, dtype=np.float64)

        result_dimensions = [SpaceDimension("Result_"+x) for x in dimensions]

        result_func = Function(name="Result_F", shape=result_shape,
                               dimensions=result_dimensions, space_order=0,
                               dtype=np.float64)

        output_grad_dimensions = [SpaceDimension(
            "output_grad"+x) for x in dimensions]

        output_grad = Function(name="outgrad_%s" % name_allocator_func(
        ), shape=result_shape, dimensions=output_grad_dimensions,
            space_order=0, dtype=np.float64)
        return (None, input_func, result_func, None, None,
                output_grad, None)

    def execute(self, input_data) -> np.array:

        self._I.data[:] = input_data
        self._R.data[:] = 0
        return super().execute()

    def equations(self):

        input_dimensions = self._I.dimensions
        res_dims = [input_dimensions[0], input_dimensions[1]]
        args = []
        for i in range(0, self._dims):
            new_dim = SpaceDimension(
                name='d_upsample'+self._dim_dict.get
                (self._dims-i, self._dims-i))
            res_dims.append(
                input_dimensions
                [-self._dims + i]*self._scale_factor[i]+new_dim)
            args.append((new_dim.name + '_M', self._scale_factor[i]-1))

        eqs = [Eq(self._R[res_dims], self._I[input_dimensions])]

        return (eqs, args)

    def backprop_equations(self, prev_layer, next_layer):
        layer = self

        if next_layer is not None:
            args = []
            next_layer_result_dim = next_layer.result_gradients.dimensions

            input_dims = [next_layer_result_dim[0], next_layer_result_dim[1]]
            eqs = [Eq(next_layer.result_gradients, 0)]
            for i in range(0, self._dims):
                new_dim = SpaceDimension(
                    name='d_kernel'+self._dim_dict.get(self._dims-i, self._dims-i))
                input_dims.append(
                    next_layer_result_dim[-self._dims + i]*self._scale_factor[i]+new_dim)
                args.append((new_dim.name + '_M', self._scale_factor[i] - 1))

            eqs += [Inc(next_layer.result_gradients[next_layer_result_dim],
                        layer.result_gradients[input_dims])]


        return (eqs, args)
