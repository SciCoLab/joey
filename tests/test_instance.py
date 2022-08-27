import torch
import pytest
from torch import nn
import joey
import numpy as np
from utils import compare


def generate_input(input_size):
    x = [x for x in input_size]
    input = torch.arange(1, 1 + np.prod(np.array(x)),
                         dtype=torch.float32).view(input_size)

    return input


def pytorch_upsample(input_data, scale_factor):

    m = torch.nn.InstanceNorm2d(input_data.shape[1])
    out = m(input_data)
    return out


@pytest.mark.parametrize("input_size,scale_factor",
                         [((2, 3, 5, 5), (2, 2)),
                          ((1, 3, 10, 48, 50), (4, 4, 5))])
def test_upsample(input_size, scale_factor):

    input_data = generate_input(input_size)

    res = pytorch_upsample(input_data, scale_factor)

    layer = joey.UpSample(input_size=input_size,
                          scale_factor=scale_factor, generate_code=True)

    joey_res = layer.execute(input_data=input_data.detach().numpy())

    compare(joey_res, res, 1e-12)


test_upsample((2, 3, 5, 5), (2, 2))
