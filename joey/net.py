import joey as ml
import numpy as np
from devito import Eq, Operator
from joey.layers import Conv
from joey.layers_v2 import ConvV2


class Net:
    """
    A class representing a neural network consisting of several layers.

    Parameters
    ----------
    layers : list of Layer
        The list of layers of a network, where the first element is an
        input layer and the last element is an output layer.

        All layers in the list should have been instantiated with
        generate_code=False.

    When instantiating this object, a C code is generated and compiled
    for both a forward and backward pass. The code is unique for the network.
    """

    def __init__(self, layers: list):
        self._layers = layers
        self._forward_arg_dict = {}
        self._backward_arg_dict = {}

        eqs, args = self._gen_eqs()
        backprop_eqs, backprop_args = self._gen_backprop_eqs()

        for (key, value) in args:
            self._forward_arg_dict[key] = value

        for (key, value) in backprop_args:
            self._backward_arg_dict[key] = value

        parameter_lists = list(map(ml.Layer.pytorch_parameters, self._layers))
        parameters = []

        for (kernel_parameter, bias_parameter) in parameter_lists:
            if kernel_parameter is not None:
                parameters.append(kernel_parameter)

            if bias_parameter is not None:
                parameters.append(bias_parameter)

        self._parameters = parameters
        self._init_parameters()

        self._forward_operator = Operator(eqs)
        self._backward_operator = Operator(backprop_eqs)

        self._forward_operator.cfunction
        self._backward_operator.cfunction

    def _init_parameters(self):
        for layer in self._layers:
            if layer.kernel is not None:
                layer.kernel.data[:] = \
                    np.random.rand(*layer.kernel.shape) - 0.5

            if layer.bias is not None:
                layer.bias.data[:] = np.random.rand(*layer.bias.shape) - 0.5

    def _gen_eqs(self):
        eqs = []
        args = []

        input_function = None

        for layer in self._layers:
            eqs.append(Eq(layer.result, 0))

        for layer in self._layers:
            if input_function is not None:
                dims = input_function.dimensions
                if isinstance(layer, ConvV2):
                    dims = list(dims)
                    for i in range(0, layer._ndims):
                        dims[2+i] = dims[2+i]+layer._padding[i]
                    eqs += [Eq(layer.input[(dims)], input_function)]
                else:
                    eqs.append(Eq(layer.input[dims], input_function[dims]))
            layer_eqs, layer_args = layer.equations()
            args += layer_args
            eqs += layer_eqs
            input_function = layer.result

        return (eqs, args)

    def _gen_backprop_eqs(self):
        eqs = []
        args = []

        for i in range(len(self._layers)):
            layer = self._layers[i]

            if layer.kernel_gradients is not None:
                eqs.append(Eq(layer.kernel_gradients, 0))

            if layer.bias_gradients is not None:
                eqs.append(Eq(layer.bias_gradients, 0))

            if layer.result_gradients is not None \
               and i < len(self._layers) - 1:
                eqs.append(Eq(layer.result_gradients, 0))

        for i in range(len(self._layers) - 1, -1, -1):
            if i < len(self._layers) - 1:
                prev_layer = self._layers[i + 1]
            else:
                prev_layer = None
                eqs += self._layers[i].activation.backprop_eqs(self._layers[i])

            if i > 0:
                next_layer = self._layers[i - 1]
            else:
                next_layer = None

            layer_eqs, layer_args = \
                self._layers[i].backprop_equations(prev_layer, next_layer)

            args += layer_args
            eqs += layer_eqs

        batch_size = self._layers[-1].result.shape[1]

        for layer in self._layers:
            if isinstance(layer, Conv) and layer.kernel_gradients is not None:
                eqs.append(Eq(layer.kernel_gradients,
                              layer.kernel_gradients / batch_size))

            if isinstance(layer, Conv) and layer.bias_gradients is not None:
                eqs.append(Eq(layer.bias_gradients,
                              layer.bias_gradients / batch_size))

        return (eqs, args)

    @property
    def pytorch_parameters(self):
        """A list of network parameters suitable for a PyTorch optimizer."""
        return self._parameters

    def forward(self, input_data):
        """
        Runs a forward pass through the network and returns its result.

        Parameters
        ----------
        input_data : np.ndarray
            Input data for the network.
        """
        inp_layer = self._layers[0]
        if isinstance(inp_layer, ConvV2):
            indices = [slice(0, inp_layer._I.shape[0], 1),
                       slice(0, inp_layer._I.shape[1], 1)]
            for i in range(inp_layer._ndims):
                indices.append(slice(inp_layer._padding[i],
                                     inp_layer._I.data.shape[2+i]
                                     - inp_layer._padding[i], 1))
            inp_layer.input.data[tuple(indices)] = input_data

        else:
            inp_layer.input.data[:] = input_data
        self._forward_operator.apply(**self._forward_arg_dict)
        return self._layers[-1].result.data

    def backward(self, expected, loss_gradient_func, pytorch_optimizer=None):
        """
        Runs a backward (backpropagation) pass through the network and
        updates the network parameters if possible.

        Parameters
        ----------
        expected : list or np.ndarray
            The list of expected results of a forward pass. It must have the
            same length as the batch size.
        loss_gradient_func : two-argument function
            A loss function to be used for a backward pass. It must accept
            two arguments: an output layer (a Layer subclass) and the list of
            expected results (see 'expected' for the details). The forward
            pass results will have been provided in the output layer through
            its 'result' property.

            The function must return either a list or a NumPy array with the
            loss values corresponding to the forward pass values (and where
            batch elements are arranged in rows rather than columns). If in
            doubt, you can have a look at examples inside the 'examples'
            directory in the Joey repository.
        pytorch_optimizer : torch.optim.Optimizer, optional
            A PyTorch optimizer that will be used for updating the network
            parameters after running a backward pass. If it's None, only
            backpropagation will be performed.

            The default value is None.
        """
        self._layers[-1].result_gradients.data[:] = \
            np.transpose(np.array(loss_gradient_func(self._layers[-1],
                                                     expected)))
        self._backward_operator.apply(**self._backward_arg_dict)

        if pytorch_optimizer is not None:
            pytorch_optimizer.step()
