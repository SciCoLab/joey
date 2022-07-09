"""
This script is used to measure the perfomance of joey 
compared with PyTorch
Image size, Iterations and batch size can be altered.
The results will be outputed in the console, together with 
the values of the parameters
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joey
import joey as ml
import numpy as np
# import matplotlib.pyplot as plt
from devito import logger, Max
import time
from performance_evaluation_PyTorch import lenet_PyTorch

from performance_evaluation_joey import lenet_Joey

logger.set_log_noperf()


# image size is altered here
image_size = 1024

# batch size is picked from here
batch_list = [1,2, 8, 16]

print("image size", image_size)
for batch_size in batch_list:
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)),
         transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
    trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    
    # itteration size is selected from here
    itter_list = [1, 2, 3, 6, 12, 25, 50]
    print("devito start time", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))

    joey_net = lenet_Joey(batch_size=batch_size,itter_list=itter_list,image_size=image_size)
    joey_layers = joey_net.train(trainloader=trainloader)

    pyTorch_net = lenet_PyTorch(batch_size=batch_size,itter_list=itter_list,image_size=image_size)
    pytorch_layers = pyTorch_net.train(trainloader=trainloader)

   
    max_error = 0
    index = -1

    for i in range(5):
        kernel = joey_layers[i].kernel.data
        pytorch_kernel = pytorch_layers[i].weight.detach().numpy()

        kernel_error = abs(kernel - pytorch_kernel) / abs(pytorch_kernel)

        bias = joey_layers[i].bias.data
        pytorch_bias = pytorch_layers[i].bias.detach().numpy()

        bias_error = abs(bias - pytorch_bias) / abs(pytorch_bias)

        error = max(np.nanmax(kernel_error), np.nanmax(bias_error))
        # print('layers[' + str(i) + '] maximum relative error: ' + str(error))

        if error > max_error:
            max_error = error
            index = i

    print()
    print('Maximum relative error is in layers[' + str(index) + ']: ' + str(max_error))
