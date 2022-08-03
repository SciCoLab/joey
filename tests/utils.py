import numpy as np
from os import environ


def compare(joey, pytorch, tolerance):
    pytorch = pytorch.detach().numpy()

    if joey.shape != pytorch.shape:
        pytorch = np.transpose(pytorch)

    assert np.allclose(joey, pytorch, atol=tolerance)


def running_in_parallel():
    if 'DEVITO_LANGUAGE' not in environ:
        return False

    return environ['DEVITO_LANGUAGE'] in ['openmp']


def get_run_count():
    if running_in_parallel():
        return 1000
    else:
        return 1
