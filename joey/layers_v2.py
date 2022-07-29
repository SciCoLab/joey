from array import array
from itertools import product
from multiprocessing.dummy import Array
from devito import Function, Eq, Inc, \
    ConditionalDimension, SpaceDimension, Operator
from sympy import And
import numpy as np
import sympy as sp
from joey import Layer
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

    def __init__(self, kernel_size, input_size, conv_dimensions,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        # Internal kernel size (self._kernel_size) is expressed as
        # (output channels / kernel count, input channels, rows, columns).
        self._conv_d = conv_dimensions

        self._error_check(kernel_size, input_size, stride, padding,
                          strict_stride_check)

        self._kernel_size = (kernel_size[0], input_size[1], *kernel_size[1:])
        self._stride = stride
        self._padding = padding

        super().__init__(self._kernel_size, input_size, activation,
                         alloc, dim_alloc,
                         generate_code)

    def _error_check(self, kernel_size, input_size, stride, padding,
                     strict_stride_check):
        if input_size is None or (len(input_size) != 4 and self._conv_d==2) \
            or (len(input_size) != 5 and self._conv_d==3) :
            raise Exception("Input size is incorrect")

        if kernel_size is None or (len(kernel_size) != 3 and self._conv_d==2) \
            or (len(kernel_size) != 4 and self._conv_d==3):
            raise Exception("Kernel size is incorrect")

        if stride is None or (len(stride) != 2  and self._conv_d==2) \
            or (len(stride) != 3 and self._conv_d==3):
            raise Exception("Stride is incorrect")

        if stride[0] < 1 or stride[1] < 1:
            raise Exception("Stride cannot be less than 1")

        if padding is None or (len(padding) != 2 and self._conv_d==2) \
            or (len(padding) != 3 and self._conv_d==3):
            raise Exception("Padding is incorrect")

        if padding[0] < 0 or padding[1] < 0:
            raise Exception("Padding cannot be negative")

        if strict_stride_check:
            if self._conv_d ==2:
                map_height = input_size[2] + 2 * padding[0]
                map_width = input_size[3] + 2 * padding[1]
                _, kernel_height, kernel_width = kernel_size

                if (map_height - kernel_height) % stride[0] != 0 or \
                (map_width - kernel_width) % stride[1] != 0:
                    raise Exception("Stride " + str(stride) + " is not "
                                    "compatible with feature map, kernel and "
                                    "padding sizes. If you want to proceed "
                                    "anyway, set strict_stride_check=False when "
                                    "instantiating this object")
            if self._conv_d ==3:
                map_height = input_size[4] + 2 * padding[0]
                map_width = input_size[3] + 2 * padding[1]
                map_depth = input_size[2] + 2 * padding[2]

                _,kernel_depth, kernel_height, kernel_width = kernel_size

                if (map_height - kernel_height) % stride[0] != 0 or \
                (map_width - kernel_width) % stride[1] != 0 or \
                (map_depth - kernel_depth) % stride[2] != 0:
                    raise Exception("Stride " + str(stride) + " is not "
                                    "compatible with feature map, kernel and "
                                    "padding sizes. If you want to proceed "
                                    "anyway, set strict_stride_check=False when "
                                    "instantiating this object")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):

        no_of_kernels = kernel_size[0]

        result_height = (input_size[-2]-kernel_size[-2] + 2 *self._padding[0])//self._stride[0] + 1

        result_width = (input_size[-1]-kernel_size[-1] + 2 *self._padding[1])//self._stride[1] + 1

        dimensions = ['dbatch', 'dchannel','dheight', 'dwidth']
        result_shape = (input_size[0], no_of_kernels, result_height,result_width)

        #modifying dims for 3d conv
        if self._conv_d == 3:
            dimensions = ['dbatch', 'dchannel', 'd_depth','dheight', 'dwidth']
            result_depth = \
                (input_size[-3]-kernel_size[-3] + 2 *self._padding[2])//self._stride[2] + 1
            result_shape = \
                (input_size[0], no_of_kernels, result_depth,result_height,result_width)


        # input data function
        input_dimensions = [SpaceDimension("Input_"+x) for x in dimensions]

        input_func = Function(name="Input_F",shape=(input_size), dimensions = input_dimensions
                ,space_order= (0,self._padding[0],self._padding[1]), dtype=np.float64)

        # function for kernel
        kernel_dims = [SpaceDimension("kernel_"+x) for x in dimensions]

        kernel_func = Function(name="Kernel_F", shape=(kernel_size)
                ,dimensions =(kernel_dims), space_order=0, dtype=np.float64)

        # Result for convolution
        result_dimensions = [SpaceDimension("Result_"+x) for x in dimensions]

        result_func = Function(name="Result_F", shape=result_shape
            , dimensions =result_dimensions, space_order=0, dtype=np.float64)

        bias_dimensions = [SpaceDimension("bias"+x) for x in ['d']]

        bias = Function(name="bias_F", shape=(kernel_size[0],)
                , dimensions =bias_dimensions, space_order=0, dtype=np.float64)

        kernel_grad_dimensions =  [SpaceDimension("kernel_grad"+x) for x in dimensions]

        kernel_grad = Function(name="kgrad_%s" % name_allocator_func(),shape=(kernel_size)
                       ,dimensions =kernel_grad_dimensions, space_order=0, dtype=np.float64)

        output_grad_dimensions = [SpaceDimension("output_grad"+x) for x in dimensions]

        output_grad = Function(name="outgrad_%s" % name_allocator_func(),shape = result_shape
                    ,dimensions = output_grad_dimensions, space_order=0, dtype=np.float64)

        bias_grad_dimensions = [SpaceDimension("bias"+x) for x in ['d']]

        bias_grad = Function(name="bgrad_%s" % name_allocator_func(),shape=(kernel_size[0],)
                , dimensions =bias_grad_dimensions, space_order=0, dtype=np.float64)

        return (kernel_func, input_func, result_func, bias, kernel_grad, output_grad, bias_grad)

    def execute(self, input_data, bias, kernel_data=None) -> np.array:
        if kernel_data is not None:
            self._K.data[:] = kernel_data

        self._bias.data[:] = bias
        self._I.data [:] = input_data
        self._R.data[:] = 0

        return super().execute()

    def equations(self):

        result_dimensions = self._R.dimensions
        bias = self._bias.dimensions

        k_height_offsets = list(range(0,self._kernel_size[-2]))
        k_width_offsets = list(range(0,self._kernel_size[-1]))

        off_sets_channels = list(range(0,self._kernel_size[1]))

        #indices of kernel matrix for convolution
        k_indices = product(off_sets_channels,k_height_offsets,k_width_offsets)

        r_height_offsets = [result_dimensions[-2]*self._stride[0]+x
                                -self._padding[0] for x in k_height_offsets]

        r_width_offsets = [result_dimensions[-1]*self._stride[1]+x
                                -self._padding[1] for x in k_width_offsets]

        #indices of input based on resullt matrix for convolution  
        r_indicies = product(off_sets_channels,r_height_offsets, r_width_offsets)

        #modifying indices for 3d convolution operation
        if len(self._conv_d)==3:
            k_depth_offsets = list(range(0,self._kernel_size[-3]))
            r_depth_offsets = [result_dimensions[-3]*self._stride[2]+x -self._padding[2]
                                for x in k_depth_offsets]
            k_indices = product(off_sets_channels,k_depth_offsets, k_height_offsets,k_width_offsets)
            r_indicies = product(off_sets_channels,r_depth_offsets,r_height_offsets,r_width_offsets)

        weight_matrix = sp.Matrix([self._K[(result_dimensions[1],*x)] for x in k_indices])

        r_indices_matrix = sp.Matrix([self._I[(result_dimensions[0],*x)] for x in r_indicies])

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
        bias_dims = layer.bias_gradients.dimensions
        dims = layer.result_gradients.dimensions

        eqs = [Inc(layer.bias_gradients[bias_dims[0]],
                   layer.result_gradients[dims[0], dims[1], dims[2], dims[3]]),
               Inc(layer.kernel_gradients[kernel_dims[0], kernel_dims[1],
                                          kernel_dims[2], kernel_dims[3]],
                   layer.result_gradients[dims[0],
                                          kernel_dims[0], dims[2],
                                          dims[3]] *
                   layer.input[dims[0], kernel_dims[1],
                               kernel_dims[2] + dims[2],
                               kernel_dims[3] + dims[3]])]

        _, _, height, width = layer.kernel.shape

        if next_layer is not None:
            next_dims = next_layer.result_gradients.dimensions
            #TODO: Better names for these dimensions
            cd1 = ConditionalDimension(name="cd_%s" % alloc(), parent=kernel_dims[2],
                                       condition=And(next_dims[2] - height +
                                                     1 + kernel_dims[2] >= 0,
                                                     next_dims[2] - height +
                                                     1 + kernel_dims[2] <
                                                     layer.result_gradients
                                                     .shape[2]))
            cd2 = ConditionalDimension(name="cd_%s" % alloc(), parent=kernel_dims[3],
                                       condition=And(next_dims[3] - width + 1 +
                                                     kernel_dims[3] >= 0,
                                                     next_dims[3] - width + 1 +
                                                     kernel_dims[3] <
                                                     layer.result_gradients
                                                     .shape[3]))

            eqs += [Inc(next_layer.result_gradients[next_dims[0],
                                                    next_dims[1],
                                                    next_dims[2],
                                                    next_dims[3]],
                        layer.kernel[dims[1], next_dims[1],
                                     height - kernel_dims[2] - 1,
                                     width - kernel_dims[3] - 1] *
                        layer.result_gradients[next_dims[0],
                                               dims[1],
                                               next_dims[2] - height + 1 +
                                               kernel_dims[2],
                                               next_dims[3] - width + 1 +
                                               kernel_dims[3]],
                        implicit_dims=(cd1, cd2))] + \
                next_layer.activation.backprop_eqs(next_layer)

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
                 stride=(1, 1,1), padding=(0, 0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        conv_dimensions = 3
        super().__init__( kernel_size, input_size,conv_dimensions,
                        stride, padding,activation, generate_code,strict_stride_check)

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
        conv_dimensions = 2
        super().__init__(kernel_size, input_size,conv_dimensions,
                        stride, padding,activation, generate_code,strict_stride_check)
