import pytest
import numpy as np
import joey
import torch

from devito import configuration


configuration['language'] = 'openmp'
# configuration['LOGGING'] = 'debug'

torch.manual_seed(0)


def pyTorch_pool3D(input_data, kernel_size, padding, stride):

    torch_conv_op = torch.nn.MaxPool3d(
        kernel_size=kernel_size, padding=padding, stride=stride)

    result = torch_conv_op(input_data)

    return result.detach().numpy()


def pyTorch_pool2D(input_data, kernel_size, padding, stride):

    torch_conv_op = torch.nn.MaxPool2d(
        kernel_size=kernel_size, padding=padding, stride=stride)

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


@pytest.mark.parametrize("input_size, kernel_size, padding, stride",
                         [((2, 3, 5, 5), (2, 3), 1, 1),
                          ((1, 3, 52, 50), (13, 13), 6, 2)])
def test_joey_pytorch_pool2d(input_size, kernel_size, padding, stride,
                             print_results=False):
    ''' test function for 3d conv operation'''
    input_data = generate_random_input(input_size)

    result_torch = pyTorch_pool2D(input_data, kernel_size, padding, stride)

    layer = joey.MaxPooling2D(kernel_size=(kernel_size),
                              input_size=(input_size), padding=(
                                  padding, padding),
                              stride=(stride, stride),
                              generate_code=True, strict_stride_check=False)
    print(layer)

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


@pytest.mark.parametrize("input_size, kernel_size, padding, stride",
                         [((2, 3, 5, 5, 9), (3, 3, 2), 1, 1),
                          ((1, 3, 5, 5, 9), (3, 3, 2), 0, 2)])
def test_joey_pytorch_pool3d(input_size, kernel_size, padding, stride,
                             print_results=False):
    ''' test function for 3d conv operation'''
    input_data = generate_random_input(input_size)

    result_torch = pyTorch_pool3D(input_data, kernel_size, padding, stride)

    layer = joey.MaxPooling3D(kernel_size=(kernel_size),
                              input_size=(input_size), padding=padding,
                              stride=stride,
                              generate_code=True, strict_stride_check=False)

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
