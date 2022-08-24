from itertools import product
import re
from webbrowser import get
from devito import Function, Eq, Inc, \
    ConditionalDimension, SpaceDimension
from sympy import And
import numpy as np
import sympy as sp
from joey import Layer
from joey import get_name 

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
        input_dimensions = [SpaceDimension(get_name("Input_"+x)) for x in dimensions]

        input_func = Function(name=get_name("Input_F"), shape=(input_size),
                              dimensions=input_dimensions, space_order=(
            self._padding[0]), dtype=np.float64)

        # function for kernel
        kernel_dims = [SpaceDimension(get_name("kernel_"+x)) for x in dimensions]

        kernel_func = Function(name=get_name("Kernel_F"), shape=(kernel_size),
                               dimensions=(kernel_dims), space_order=0,
                               dtype=np.float64)

        # Result for convolution
        result_dimensions = [SpaceDimension(get_name("Result_"+x)) for x in dimensions]

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

        bias_grad_dimensions = [SpaceDimension(get_name("bias_"+x)) for x in ['d']]

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
                list(range(result_grad_shape[-self._dims + i]-1,-1,-1)))

        # indices of kernel matrix for convolution
        k_indices = product(* k_dims_offsets)

        r_dims_offsets = []

        # generating offsets in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on
      
        for i in range(0, self._dims):
            r_dim_offsets = [kernel_dims[-self._dims + i] +x*self._stride[i] -self._padding[i] for x in k_dims_offsets[i]]
            r_dims_offsets.append(r_dim_offsets)

        # indices of input based on resullt matrix for convolution
        r_indicies = product(*r_dims_offsets)

        weight_matrix = sp.Matrix(
            [layer.result_gradients[(*result_grad_dims[0:1],kernel_dims[0], *x)] for x in k_indices])

        r_indices_matrix = sp.Matrix(
            [self._I[(result_grad_dims[0],kernel_dims[1], *x)] for x in r_indicies])

        # stencil operation corresponding to the convolution
        sten = weight_matrix.dot(r_indices_matrix)
        eqs = [Inc(layer.bias_gradients[bias_dims[0]],
                  kernel_shape[0]* layer.result_gradients[result_grad_dims])]
        eqs+=  [ Inc(layer.kernel_gradients[(kernel_dims)], kernel_shape[0]* sten)]


        if next_layer is not None:
            next_layer_dims = next_layer.result_gradients.dimensions

            k_dims_offsets = []
            for i in range(0, self._dims):
                k_dims_offsets.append(
                    list(range(kernel_shape[-self._dims + i]-1,-1,-1)))
            off_sets_channels = list(range(0, self._kernel_size[1]))

            # indices of kernel matrix for convolution
            k_indices = product(off_sets_channels, * k_dims_offsets)


            k_dims_offsets = []
            for i in range(0, self._dims):
                k_dims_offsets.append(
                    list(range(0,kernel_shape[-self._dims + i])))
            r_dims_offsets = []

            # generating offsets in the order depth, height, width ,
            # hence arr[-3], arr[-2] and so on
        
            for i in range(0, self._dims):
                r_dim_offsets = [next_layer_dims[-self._dims + i] +x*(self._stride[i]-1)  for x in k_dims_offsets[i]]
                r_dims_offsets.append(r_dim_offsets)

            # indices of input based on resullt matrix for convolution
            r_indicies = product(off_sets_channels, *r_dims_offsets)

            weight_matrix = sp.Matrix(
                [self._K[(result_grad_dims[1], *x)] for x in k_indices])

            r_indices_matrix = sp.Matrix(
                [layer.result_gradients[(next_layer_dims[0], *x)] for x in r_indicies])

            # stencil operation corresponding to the convolution
            sten = weight_matrix.dot(r_indices_matrix)
            
            eqs+=  [ Eq(next_layer.result_gradients[(next_layer_dims)], sten)]
            
        # if next_layer is not None:
        #     next_dims = next_layer.result_gradients.dimensions
        #     # TODO: Better names for these dimensions
        #     cd1 = ConditionalDimension(name="cd_%s" % alloc(),
        #                                parent=kernel_dims[2],
        #                                condition=And(next_dims[2] - height +
        #                                              1 + kernel_dims[2] >= 0,
        #                                              next_dims[2] - height +
        #                                              1 + kernel_dims[2] <
        #                                              layer.result_gradients
        #                                              .shape[2]))
        #     cd2 = ConditionalDimension(name="cd_%s" % alloc(),
        #                                parent=kernel_dims[3],
        #                                condition=And(next_dims[3] - width + 1 +
        #                                              kernel_dims[3] >= 0,
        #                                              next_dims[3] - width + 1 +
        #                                              kernel_dims[3] <
        #                                              layer.result_gradients
        #                                              .shape[3]))

        #     eqs += [Inc(next_layer.result_gradients[next_dims[0],
        #                                             next_dims[1],
        #                                             next_dims[2],
        #                                             next_dims[3]],
        #                 layer.kernel[dims[1], next_dims[1],
        #                              height - kernel_dims[2] - 1,
        #                              width - kernel_dims[3] - 1] *
        #                 layer.result_gradients[next_dims[0],
        #                                        dims[1],
        #                                        next_dims[2] - height + 1 +
        #                                        kernel_dims[2],
        #                                        next_dims[3] - width + 1 +
        #                                        kernel_dims[3]],
        #                 implicit_dims=(cd1, cd2))] + \
        #         next_layer.activation.backprop_eqs(next_layer)

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
