import joey
import joey as ml
from devito import logger, Max
import numpy as np
import time
import torch.optim as optim

class lenet_Joey():

    def __init__(self,image_size, itter_list, batch_size):

        self.image_size = image_size
        self.itter_list = itter_list
        self.batch_size = batch_size
        self.devito_net, self.devito_layers = self.create_lenet()
        self.optimizer = optim.SGD(self.devito_net.pytorch_parameters, lr=0.001, momentum=0.9)



    def create_lenet(self):
        # Six 3x3 filters, activation RELU
        layer1 = ml.Conv(kernel_size=(6, 3, 3),
                        input_size=(self.batch_size, 1, self.image_size , self.image_size),
                        activation=ml.activation.ReLU(),
                        generate_code=False)
        # Max 2x2 subsampling
        layer2 = ml.MaxPooling(kernel_size=(2, 2),
                            input_size=(self.batch_size, 6, self.image_size - 2, self.image_size - 2),
                            stride=(2, 2),
                            generate_code=False)
        # Sixteen 3x3 filters, activation RELU
        layer3 = ml.Conv(kernel_size=(16, 3, 3),
                        input_size=(self.batch_size, 6, (self.image_size - 2) // 2, (self.image_size - 2) // 2),
                        activation=ml.activation.ReLU(),
                        generate_code=False)
        # Max 2x2 subsampling
        layer4 = ml.MaxPooling(kernel_size=(2, 2),
                            input_size=(self.batch_size, 16, ((self.image_size - 2) // 2) - 2, ((self.image_size - 2) // 2) - 2),
                            stride=(2, 2),
                            strict_stride_check=False,
                            generate_code=False)
        # Full connection (16 * 6 * 6 -> 120), activation RELU
        pooled_size = (((self.image_size - 2) // 2) - 2) // 2
        layer5 = ml.FullyConnected(weight_size=(120, 16 * pooled_size * pooled_size),
                                input_size=(16 * pooled_size * pooled_size, self.batch_size),
                                activation=ml.activation.ReLU(),
                                generate_code=False)
        # Full connection (120 -> 84), activation RELU
        layer6 = ml.FullyConnected(weight_size=(84, 120),
                                input_size=(120, self.batch_size),
                                activation=ml.activation.ReLU(),
                                generate_code=False)
        # Full connection (84 -> 10), output layer
        layer7 = ml.FullyConnectedSoftmax(weight_size=(10, 84),
                                        input_size=(84, self.batch_size),
                                        generate_code=False)
        # Flattening layer necessary between layer 4 and 5
        layer_flat = ml.Flat(input_size=(self.batch_size, 16, pooled_size, pooled_size),
                            generate_code=False)

        layers = [layer1, layer2, layer3, layer4,
                layer_flat, layer5, layer6, layer7]

        return (ml.Net(layers), layers)
    def train(self,trainloader):
        for itter in self.itter_list:
            start_time = time.time()
            for i, data in enumerate(trainloader, 0):
                images, labels = data
                images.double()

                self.train_per_data(images, labels, self.optimizer)

                if i == itter - 1:
                    break
            elapsed_time = time.time() - start_time
            print("batch :", self.batch_size, "itterations:", itter, "devito: ", elapsed_time)
        return [self.devito_layers[0], self.devito_layers[2], self.devito_layers[5], self.devito_layers[6], self.devito_layers[7]]



    def train_per_data(self, input_data, expected_results, pytorch_optimizer):
        self.devito_net.forward(input_data)

        def loss_grad(layer, expected):
            gradients = []

            # revisit this part, needs to be generalized better
            if self.batch_size == 1:
                y_pred = layer.result.data
                y = expected
                return [np.log2(np.exp(y_pred[int(y)]) / (np.sum(np.exp(y_pred))))]

            for b in range(self.batch_size):
                y_pred = layer.result.data

                y = expected[b]

                row = np.log2(np.exp(y_pred[int(y)][b]) / (np.sum(np.exp(y_pred[:, b]))))

                gradients.append(row)

            return gradients

        self.devito_net.backward(expected_results, loss_grad, pytorch_optimizer)


    def relu(self,x):
        return Max(0, x)


    def maximum(self,lst):
        return Max(*lst)

