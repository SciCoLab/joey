import pytest
import numpy as np
import joey
import torch

from devito import configuration


configuration['language'] = 'openmp'
# configuration['LOGGING'] = 'debug'

torch.manual_seed(0)


def pyTorch_norm3D(input_data):

    torch_conv_op = torch.nn.InstanceNorm3d(input_data.shape[1])

    result = torch_conv_op(input_data)

    return result.detach().numpy()


def pyTorch_norm2D(input_data):

    torch_conv_op = torch.nn.InstanceNorm2d(input_data.shape[1])

    result = torch_conv_op(input_data)

    return result.detach().numpy()


def generate_random_input(input_size) -> tuple:
    '''generate random data for test'''

    input_data = \
        torch.randn(input_size[0], input_size[1],
                    input_size[-2], input_size[-1], dtype=torch.float64)
    if len(input_size) == 5:
        input_data = \
            torch.randn(input_size[0], input_size[1], input_size[2],
                        input_size[3], input_size[4], dtype=torch.float64)

    return input_data


@pytest.mark.parametrize("input_size",
                         [((2, 3, 5, 5)),((1, 3, 52, 50))])
def test_joey_pytorch_norm2d(input_size,print_results=False):
    ''' test function for 3d conv operation'''
    input_data = generate_random_input(input_size)

    result_torch = pyTorch_norm2D(input_data)

    layer = joey.InstanceNorm2D(input_size=(input_size),
                              generate_code=True)

    input_numpy = input_data.detach().numpy()

    mean = np.sum(input_numpy)/25
    print(mean)
    input_mean = input_numpy - mean
    x = input_mean*input_mean
    x = np.sum(x)/25
    print("hi", (x))
    result_joey = layer.execute(input_numpy)
    print(layer._mean.data)
    print(layer._var.data)

    if print_results:
        print("torch", result_torch)

        print("joey", result_joey)
        match = np.allclose(result_joey, result_torch)
        print("Do they match \n", match)

        if not match:
            
            print("\n", np.abs(result_joey)-np.abs(result_torch))

    # assert np.allclose(result_joey, result_torch)


@pytest.mark.parametrize("input_size",
                         [((2, 3, 5, 5, 9)),((1, 3, 5, 5, 9))])
def test_joey_pytorch_norm3d(input_size, print_results=False):
    ''' test function for 3d conv operation'''
    input_data = generate_random_input(input_size)

    result_torch = pyTorch_norm3D(input_data)
    layer = joey.InstanceNorm3D(input_size=(input_size),
                              generate_code=True)

    input_numpy = input_data.detach().numpy()
    result_joey = layer.execute(input_numpy)

    if print_results:
        print("torch", result_torch)

        print("joey", result_joey)
    match = np.allclose(result_joey, result_torch)
    print("Do they match \n", match)

    if not match:
        print("\n", np.abs(result_joey)-np.abs(result_torch))

    assert np.allclose(result_joey, result_torch)


test_joey_pytorch_norm2d((1,1, 5, 5), True)