from devito import Function, Eq, Inc, SpaceDimension
import numpy as np
from joey import Layer
from joey import default_name_allocator as alloc


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

        self._ndims = dimensions

        if (type(padding) is int):
            padding = tuple([padding] * self._ndims)
        if (type(stride) is int):
            stride = tuple([stride] * self._ndims)
        self._error_check(kernel_size, input_size, stride, padding,
                          strict_stride_check)

        # Internal kernel size (self._kernel_size) is expressed as
        # (output channels / kernel count, input channels, rows, columns).
        self._kernel_size = (kernel_size[0], input_size[1], *kernel_size[1:])
        self._stride = stride
        self._padding = padding

        super().__init__(self._kernel_size, input_size, activation,
                         alloc, self.get_name,
                         generate_code)

    def _error_check(self, kernel_size, input_size, stride, padding,
                     strict_stride_check):
        if input_size is None or (len(input_size) != self._ndims+2):
            raise Exception("Input size is incorrect")

        if kernel_size is None or (len(kernel_size) != self._ndims+1):
            raise Exception("Kernel size is incorrect")

        if stride is None or (len(stride) != self._ndims):
            raise Exception("Stride is incorrect")

        if padding is None or (len(padding) != self._ndims):
            raise Exception("Padding is incorrect")

        for i in range(0, self._ndims):

            if (type(stride[i]) is not int):
                raise Exception("Stride must be an integer")

            if (type(padding[i]) is not int):
                raise Exception("Padding must be an integer")

            if stride[i] < 1:
                raise Exception("Stride cannot be less than 1")

            if padding[i] < 0:
                raise Exception("Padding cannot be negative")

        if strict_stride_check:
            input_d = input_size[-self._ndims+i] + 2 * padding[i]
            if (input_d - kernel_size[-self._ndims+i]) % stride[i] != 0:
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
        input_size = list(input_size)
        # generating  in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on
        for i in range(0, self._ndims):
            result_d = (input_size[(-self._ndims+i)] -
                        kernel_size[(-self._ndims+i)] +
                        2 * self._padding[i])//self._stride[i] + 1
            input_size[(-self._ndims+i)] += 2 * self._padding[i]

            result_shape.append(result_d)
            dimensions.append('d_'+dim_dict.get(self._ndims-i, self._ndims-i))

        result_shape = (input_size[0], no_of_kernels, *result_shape)

        # input data function
        input_dimensions = [SpaceDimension(
            get_name("Input_"+x)) for x in dimensions]

        input_func = Function(name=get_name("Input_F"), shape=(input_size),
                              dimensions=input_dimensions, space_order=0,
                              dtype=np.float64)

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

        kernel_grad = Function(name=get_name("kgrad_"), shape=(
            kernel_size), dimensions=kernel_grad_dimensions, space_order=0,
            dtype=np.float64)

        output_grad_dimensions = [SpaceDimension(
            get_name("output_grad"+x)) for x in dimensions]
        self._output_grad_padded_dimensions = [SpaceDimension(
            get_name("output_grad_padded"+x)) for x in dimensions]
        self._output_grad_dilated_dims = [SpaceDimension(
            get_name("output_dil"+x)) for x in dimensions]

        output_grad = Function(name=get_name("outgrad_"),
                               shape=result_shape,
                               dimensions=output_grad_dimensions,
                               space_order=0, dtype=np.float64)

        bias_grad_dimensions = [SpaceDimension(
            get_name("bias_"+x)) for x in ['d']]

        bias_grad = Function(name=get_name("bgrad_"), shape=(
            kernel_size[0],), dimensions=bias_grad_dimensions, space_order=0,
            dtype=np.float64)

        return (kernel_func, input_func, result_func, bias, kernel_grad,
                output_grad, bias_grad)

    def execute(self, input_data, bias, kernel_data=None) -> np.array:
        if kernel_data is not None:
            self._K.data[:] = kernel_data

        self._bias.data[:] = bias
        self._R.data[:] = 0
        indices = [slice(0, self._I.shape[0], 1),
                   slice(0, self._I.shape[1], 1)]
        for i in range(self._ndims):
            indices.append(slice(self._padding[i],
                                 self._I.data.shape[2+i]-self._padding[i], 1))

        self._I.data[tuple(indices)] = input_data
        return super().execute()

    def equations(self):

        result_dimensions = self._R.dimensions
        kernel_dims = self._K.dimensions
        eqs = []
        input_dims = [result_dimensions[0], kernel_dims[1]]
        for i in range(0, self._ndims):
            input_dims.append(
                result_dimensions[-self._ndims + i]*self._stride[i] +
                kernel_dims[-self._ndims + i])
        eqs += [Inc(self._R[result_dimensions],
                    self._K[(result_dimensions[1],
                             *kernel_dims[1:])]*self._I[input_dims])]

        eqs.append(Inc(self._R[result_dimensions],
                   self._bias[result_dimensions[1]]))
        if self._activation is not None:
            eqs.append(Eq(self._R[result_dimensions],
                          self._activation(self._R[result_dimensions])))
        return (eqs, [])

    def backprop_equations(self, prev_layer, next_layer):
        layer = self

        kernel_dims = layer.kernel_gradients.dimensions
        bias_dims = layer.bias_gradients.dimensions
        result_grad_dims = layer.result_gradients.dimensions
        result_grad_shape = layer.result_gradients.shape

        input_dims = [result_grad_dims[0], kernel_dims[1]]
        eqs = []
        padded_shape = [0] * self._ndims
        for i in range(0, self._ndims):
            padded_shape[-self._ndims+i] = result_grad_shape[-self._ndims+i] \
                + (self._stride[0]-1)*(result_grad_shape[-self._ndims+i]-1)

        res_grad_dilated = Function(name=self.get_name("outgrad_dilated"),
                                    shape=(*result_grad_shape[0:2],
                                           *padded_shape),
                                    dimensions=self._output_grad_dilated_dims,
                                    space_order=(0),
                                    dtype=np.float64)
        dims = list(result_grad_dims)
        for i in range(0, layer._ndims):
            dims[2+i] = dims[2+i] + (dims[2+i] * (self._stride[i]-1))
        eqs += [Eq(res_grad_dilated[(dims)], layer.result_gradients)]

        for i in range(0, self._ndims):
            input_dims.append(
                kernel_dims[-self._ndims + i] +
                self._output_grad_dilated_dims[-self._ndims + i])

        eqs += [Inc(layer.kernel_gradients[kernel_dims],

                    res_grad_dilated[(result_grad_dims[0], kernel_dims[0],
                                      *self._output_grad_dilated_dims[2:]
                                      )]*self._I[input_dims])]

        eqs += [Inc(layer.bias_gradients[bias_dims],
                    layer.result_gradients[(result_grad_dims[0], bias_dims[0],
                                            *result_grad_dims[2:])])]

        if next_layer is not None:
            next_layer_dims = next_layer.result_gradients.dimensions
            padded_shape = [0] * self._ndims
            for i in range(0, self._ndims):
                padded_shape[-self._ndims+i] = result_grad_shape[
                    -self._ndims+i]\
                    + (2*(self._padding[0]+self._kernel_size[-1]-1)) + \
                    (self._stride[0]-1)*(result_grad_shape[-self._ndims+i]-1)
            self.op = output_grad_padded = Function(
                name=self.get_name("outgrad_padded"),
                shape=(*result_grad_shape[0:2], *padded_shape),
                dimensions=self._output_grad_padded_dimensions,
                space_order=(0), dtype=np.float64)
            dims = list(result_grad_dims)
            for i in range(0, layer._ndims):
                dims[2+i] = dims[2+i] - layer._padding[i] + \
                    (self._kernel_size[-1]-1) + dims[2+i] * (self._stride[0]-1)
            eqs += [Eq(output_grad_padded[(dims)], layer.result_gradients)]

            input_dims = [next_layer_dims[0], kernel_dims[0]]
            reversed_k_dims = [kernel_dims[0],
                               next_layer_dims[1], *kernel_dims[2:]]
            for i in range(0, self._ndims):
                input_dims.append(
                    next_layer_dims[-self._ndims + i] +
                    kernel_dims[-self._ndims + i])
                reversed_k_dims[-self._ndims+i] = \
                    reversed_k_dims[-self._ndims
                                    + i].symbolic_max - \
                    reversed_k_dims[-self._ndims+i]
            eqs += [Inc(next_layer.result_gradients[next_layer_dims],
                        self._K[(reversed_k_dims)]
                        * output_grad_padded[input_dims], implicit_dims=(
                            next_layer_dims[0], kernel_dims[0]))]

            eqs += next_layer.activation.backprop_eqs(next_layer)

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
