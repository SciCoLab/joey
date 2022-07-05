"""
Additional  Joey Layers
The code was written originally here, with its version history:
https://github.com/cha-tzi/devito/blob/ml/devito/ml/layers.py

"""
from joey import Layer
from joey import default_name_allocator as alloc
from joey import default_dim_allocator as dim_alloc
from devito import Grid, Function, Eq, Inc
import numpy as np

# Mathematically, TransConv uses a fractionally-strided convolution
class TransConv(Layer):
    """
    A new class to store the Transposed Convolution Operator
    It is based on the Conv class.
    The diffference is that new parametersz and p’ are calculated.
    Between each row and column of the input tensor a z number
    of zeros is added, and the output is padded by the p’ amount
    """
    def __init__(self, kernel_size, input_size,
                 name_allocator_func=alloc, dim_allocator_func=dim_alloc,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=True,
                 strict_stride_check=True):
        # Kernel size argument (kernel_size) is expressed as
        # (output channels / kernel count, rows, columns).
        # Internal kernel size (self._kernel_size) is expressed as
        # (output channels / kernel count, input channels, rows, columns).
        # Input size is expressed as (batch size, channels, rows, columns).

        self._error_check(kernel_size, input_size, stride, padding,
                        strict_stride_check)

        self._kernel_size = (kernel_size[0], input_size[1], kernel_size[1],
                             kernel_size[2])
        self._activation = activation
        self._stride = stride
        self._padding = padding
        # z : number of zeros to be inserted btw each cols and rows
        self._z = self._stride[0] - 1
        # transpad : number of 0s to be inserted around the image
        self._transpadding = (self._kernel_size[2] - self._padding[0] - 1,
                              self._kernel_size[3] - self._padding[1] - 1)

        super().__init__(self._kernel_size, input_size, name_allocator_func,
                         dim_allocator_func, generate_code)

    def _error_check(self, kernel_size, input_size, stride, padding,
                     strict_stride_check):
        if input_size is None or len(input_size) != 4:
            raise Exception("Input size is incorrect")

        if kernel_size is None or len(kernel_size) != 3:
            raise Exception("Kernel size is incorrect")

        if stride is None or len(stride) != 2:
            raise Exception("Stride is incorrect")

        if stride[0] < 1 or stride[1] < 1:
            raise Exception("Stride cannot be less than 1")

        if padding is None or len(padding) != 2:
            raise Exception("Padding is incorrect")

        if padding[0] < 0 or padding[1] < 0:
            raise Exception("Padding cannot be negative")

        if strict_stride_check:
            _, kernel_height, kernel_width = kernel_size
            # we need to create a new map height based on the 
            # new padding
            transpadding = (kernel_height - padding[0] - 1,
                            kernel_width - padding[1] - 1)
            map_height = 2*input_size[2] - 1 + 2 * transpadding[0]
            map_width = 2*input_size[3] - 1 + 2 * transpadding[1]
            

            if (map_height - kernel_height) % 1 != 0 or \
               (map_width - kernel_width) % 1 != 0:
                raise Exception("Stride " + str(stride) + " is not "
                                "compatible with feature map, kernel and "
                                "padding sizes")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):
        """
        Function to  allocate enough space for data to be inserted
        into the devito grids
        """
        map_height = 2*input_size[2] -1 + 2 * self._transpadding[0]
        map_width = 2*input_size[3] -1 + 2 * self._transpadding[1]
        _, _, kernel_height, kernel_width = kernel_size

        gridK = Grid(shape=kernel_size, dimensions=dim_allocator_func(4))
        K = Function(name=name_allocator_func(), grid=gridK, space_order=0,
                     dtype=np.float64)

        gridB = Grid(shape=(input_size[0], input_size[1],
                            map_height, map_width),
                     dimensions=dim_allocator_func(4))
        B = Function(name=name_allocator_func(), grid=gridB, space_order=0,
                     dtype=np.float64)

        gridR = Grid(shape=(input_size[0], kernel_size[0],
                            (input_size[2]- 1) * self._stride[0] \
                            + kernel_height - 2 * self._padding[0],
                            (input_size[3]- 1) * self._stride[1] \
                            + kernel_width - 2 * self._padding[1]),
                     dimensions=dim_allocator_func(4))
        R = Function(name=name_allocator_func(), grid=gridR, space_order=0,
                     dtype=np.float64)

        bias_grid = Grid(shape=kernel_size[0],
                         dimensions=dim_allocator_func(1))
        bias = Function(name=name_allocator_func(), grid=bias_grid,
                        space_order=0, dtype=np.float64)

        kernel_grad = Function(name=name_allocator_func(),
                               grid=gridK, space_order=0, dtype=np.float64)

        output_grad = Function(name=name_allocator_func(),
                               grid=Grid(shape=(gridR.shape[1],
                                                gridR.shape[2],
                                                gridR.shape[3]),
                                         dimensions=dim_allocator_func(3)),
                               space_order=0, dtype=np.float64)

        bias_grad = Function(name=name_allocator_func(),
                             grid=bias_grid, space_order=0, dtype=np.float64)

        return (K, B, R, bias, kernel_grad, output_grad, bias_grad)

    def execute(self, input_data, bias, kernel_data=None):
        map_height = 2*input_data.shape[2] -1 + 2 * self._transpadding[0]
        batch_size, channels, _, _ = input_data.shape
        #init the zeros
        out = np.zeros((batch_size, channels, (self._z+1)*input_data.shape[2]-1,
                        (self._z+1)*input_data.shape[3]-1))
        out[..., ::self._z+1, ::self._z+1] = input_data
        for i in range(batch_size):
            for j in range(channels):
                for k in range(self._transpadding[0],
                               map_height - self._transpadding[0]):
                    self._I.data[i, j, k] = \
                        np.concatenate(([0] * self._transpadding[1],
                                        out[i, j, k - self._transpadding[0]],
                                        [0] * self._transpadding[1]))
        if kernel_data is not None:
            self._K.data[:] = kernel_data

        self._bias.data[:] = bias

        self._R.data[:] = 0

        return super().execute()

    def equations(self, input_function=None):
        if input_function is None:
            input_function = self._I

        a, b, c, d = self._R.dimensions
        _, _, kernel_height, kernel_width = self._kernel_size
        batch_size, channels, _, _ = input_function.shape
        e, f, g, h = self._K.dimensions
        #stide is always 1 in trans-conv
        #print("shape of I", self._I.shape)
        rhs = sum([self._K[e, f, x, y] *
                   input_function[a, f, c + x, d + y]
                   for x in range(kernel_height)
                   for y in range(kernel_width)])

        eqs = [Inc(self._R[a, e, c, d], rhs),
               Inc(self._R[a, e, c, d], self._bias[e])]

        if self._activation is not None:
            eqs.append(Eq(self._R[a, e, c, d],
                          self._activation(self._R[a, e, c, d])))

        return eqs

class BatchNorm(Layer):
    def __init__(self, input_size, eps=1e-05,
                 name_allocator_func=alloc, dim_allocator_func=dim_alloc,
                 generate_code=True):

        self._eps = eps

        super().__init__(input_size, self._eps, name_allocator_func,
                         dim_allocator_func, generate_code)

    def _allocate(self, input_size, eps, name_allocator_func,
                  dim_allocator_func):
        """
        Function to  allocate enough space for data to be inserted
        into the devito grids
        """
        a, b, c, d = dim_allocator_func(4)
        gridI = Grid(shape=(input_size[0], input_size[1], input_size[2], input_size[3]),
         dimensions=(a, b, c, d))
        I = Function(name=name_allocator_func(), grid=gridI, space_order=0,
                     dtype=np.float64)
        
        e, f, g, h = dim_allocator_func(4)
        gridR = Grid(shape=(input_size), dimensions=(e, f, g, h))

        R = Function(name=name_allocator_func(), grid=gridR, space_order=0,
                     dtype=np.float64)

        _, channels, _, _ = input_size
        gridM = Grid(shape=(channels))
        M = Function(name=name_allocator_func(), grid=gridM, space_order=0,
                     dtype=np.float64)
        
        gridV = Grid(shape=(channels))
        V = Function(name=name_allocator_func(), grid=gridV, space_order=0,
                     dtype=np.float64)
        return(None, I, R, M, V, None, None)

    def execute(self, input_data):
        images, channels, height, width = input_data.shape
        means = np.zeros(channels)

        for channel in range(channels):
            for image in range(images):
                means[channel] += sum(sum(input_data[image][channel]))/(height*width)
        means /= images
        M.data[:] = means

        var = np.zeros(channels)
        for channel in range(channels):
            for image in range(images):
                var[channel] += sum(sum(pow(input_data[image][channel]-means[channel],2)))/(height*width)
                    
        var /= images
        V.data[:] = var
        return super().execute()

    def equations(self, input_function=None):
        a, b, c, d = self._R.dimensions
        rhs = ((self._I[a, b, c, d]-self._M[b])/(((self._R[b]+self._epsilon)**(0.5))))
        return [Eq(self._R[a, b, c, d], rhs)]
