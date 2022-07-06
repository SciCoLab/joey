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
from torch._C._autograd import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile

import joey
import joey as ml
import numpy as np
# import matplotlib.pyplot as plt
from devito import logger, Max
import time

logger.set_log_noperf()


def create_lenet():

    # Six 3x3 filters, activation RELU
    layer1 = ml.Conv(kernel_size=(6, 3, 3),
                     input_size=(batch_size, 1, image_size, image_size),
                     activation=ml.activation.ReLU(),
                     generate_code=False)
    # Max 2x2 subsampling
    layer2 = ml.MaxPooling(kernel_size=(2, 2),
                           input_size=(batch_size, 6, image_size - 2, image_size - 2),
                           stride=(2, 2),
                           generate_code=False)
    # Sixteen 3x3 filters, activation RELU
    layer3 = ml.Conv(kernel_size=(16, 3, 3),
                     input_size=(batch_size, 6, (image_size - 2) // 2, (image_size - 2) // 2),
                     activation=ml.activation.ReLU(),
                     generate_code=False)
    # Max 2x2 subsampling
    layer4 = ml.MaxPooling(kernel_size=(2, 2),
                           input_size=(batch_size, 16, ((image_size - 2) // 2) - 2, ((image_size - 2) // 2) - 2),
                           stride=(2, 2),
                           strict_stride_check=False,
                           generate_code=False)
    # Full connection (16 * 6 * 6 -> 120), activation RELU
    pooled_size = (((image_size - 2) // 2) - 2) // 2
    layer5 = ml.FullyConnected(weight_size=(120, 16 * pooled_size * pooled_size),
                               input_size=(16 * pooled_size * pooled_size, batch_size),
                               activation=ml.activation.ReLU(),
                               generate_code=False)
    # Full connection (120 -> 84), activation RELU
    layer6 = ml.FullyConnected(weight_size=(84, 120),
                               input_size=(120, batch_size),
                               activation=ml.activation.ReLU(),
                               generate_code=False)
    # Full connection (84 -> 10), output layer
    layer7 = ml.FullyConnectedSoftmax(weight_size=(10, 84),
                                      input_size=(84, batch_size),
                                      generate_code=False)
    # Flattening layer necessary between layer 4 and 5
    layer_flat = ml.Flat(input_size=(batch_size, 16, pooled_size, pooled_size),
                         generate_code=False)

    layers = [layer1, layer2, layer3, layer4,
              layer_flat, layer5, layer6, layer7]

    return (ml.Net(layers), layers)


def relu(x):
    return Max(0, x)


def maximum(lst):
    return Max(*lst)


def train(devito_Net, input_data, expected_results, pytorch_optimizer):
    devito_Net.forward(input_data)

    def loss_grad(layer, expected):
        gradients = []

        # revisit this part, needs to be generalized better
        if batch_size == 1:
            y_pred = layer.result.data
            y = expected
            return [np.log2(np.exp(y_pred[int(y)]) / (np.sum(np.exp(y_pred))))]

        for b in range(batch_size):
            y_pred = layer.result.data

            y = expected[b]

            row = np.log2(np.exp(y_pred[int(y)][b]) / (np.sum(np.exp(y_pred[:, b]))))

            gradients.append(row)

        return gradients

    devito_Net.backward(expected_results, loss_grad, pytorch_optimizer)


# image size is altered here
image_size = 980

# batch size is picked from here
batch_list = [1, 2, 8, 16]

print("image size", image_size)
for batch_size in batch_list:
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)),
         transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
    trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    devito_net, devito_layers = create_lenet()
    optimizer = optim.SGD(devito_net.pytorch_parameters, lr=0.001, momentum=0.9)

    layer1_kernel = torch.tensor(devito_layers[0].kernel.data)
    layer1_bias = torch.tensor(devito_layers[0].bias.data)
    layer3_kernel = torch.tensor(devito_layers[2].kernel.data)
    layer3_bias = torch.tensor(devito_layers[2].bias.data)
    layer5_kernel = torch.tensor(devito_layers[5].kernel.data)
    layer5_bias = torch.tensor(devito_layers[5].bias.data)
    layer6_kernel = torch.tensor(devito_layers[6].kernel.data)
    layer6_bias = torch.tensor(devito_layers[6].bias.data)
    layer7_kernel = torch.tensor(devito_layers[7].kernel.data)
    layer7_bias = torch.tensor(devito_layers[7].bias.data)

    # itteration size is selected from here
    itter_list = [1, 2, 3, 6, 12, 25, 50]
    print("devito start time", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))

    for itter in itter_list:

        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images.double()

            train(devito_net, images, labels, optimizer)

            if i == itter - 1:
                break
        elapsed_time = time.time() - start_time
        print("batch :", batch_size, "itterations:", itter, "devito: ", elapsed_time)


    # PyTorch:

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
            pooled_size = (((image_size - 2) // 2) - 2) // 2
            self.fc1 = nn.Linear(16 * pooled_size * pooled_size, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]
            num_features = 1
            for s in size:
                num_features *= s
            return num_features


    net = Net()
    net.double()

    with torch.no_grad():
        net.conv1.weight[:] = layer1_kernel
        net.conv1.bias[:] = layer1_bias
        net.conv2.weight[:] = layer3_kernel
        net.conv2.bias[:] = layer3_bias
        net.fc1.weight[:] = layer5_kernel
        net.fc1.bias[:] = layer5_bias
        net.fc2.weight[:] = layer6_kernel
        net.fc2.bias[:] = layer6_bias
        net.fc3.weight[:] = layer7_kernel
        net.fc3.bias[:] = layer7_bias

    start_time = time.time()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print("pyTorch start time", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))
    for itter in itter_list:
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.double())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i==1:
                with profile(activities=[ProfilerActivity.CPU],
                             profile_memory=True, record_shapes=True) as prof:
                    net(images.double())
                print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

            if i == itter - 1:
                break
        elapsed_time = time.time() - start_time
        print("batch :", batch_size, "itterations:", itter, "pytorch: ", elapsed_time)

    layers = [devito_layers[0], devito_layers[2], devito_layers[5], devito_layers[6], devito_layers[7]]
    pytorch_layers = [net.conv1, net.conv2, net.fc1, net.fc2, net.fc3]

    max_error = 0
    index = -1

    for i in range(5):
        kernel = layers[i].kernel.data
        pytorch_kernel = pytorch_layers[i].weight.detach().numpy()

        kernel_error = abs(kernel - pytorch_kernel) / abs(pytorch_kernel)

        bias = layers[i].bias.data
        pytorch_bias = pytorch_layers[i].bias.detach().numpy()

        bias_error = abs(bias - pytorch_bias) / abs(pytorch_bias)

        error = max(np.nanmax(kernel_error), np.nanmax(bias_error))
        # print('layers[' + str(i) + '] maximum relative error: ' + str(error))

        if error > max_error:
            max_error = error
            index = i

    print()
    print('Maximum relative error is in layers[' + str(index) + ']: ' + str(max_error))
