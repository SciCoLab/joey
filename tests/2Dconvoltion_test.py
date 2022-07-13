from bisect import bisect
from calendar import c
import weakref
import numpy as np
import torch

import joey as ml
from joey.net import Net
from scipy import signal
image_size = 1024
input_data = torch.rand(1,1,image_size, image_size)

input  = []
c=0;
for i in range(0,5):
    temp =[]
    for j in range(0,5):
        temp.append(c)
        c = c+1;
    input.append(temp)
custom_kernel = [[0,0,0] ,[1,0,-1],[0,0,0]]

weights = torch.randn(1,1, 3, 3)
weights[0][0] = torch.Tensor(custom_kernel)

custom_input_T = torch.randn(1, 1, 5, 5)


custom_input_T[0][0]= torch.Tensor(input);


torch_conv_op = torch.nn.Conv2d(1,1, kernel_size=3, stride=1)



with torch.no_grad():
    torch_conv_op.weight = torch.nn.Parameter(weights)
    torch_conv_op.bias = torch.nn.Parameter(torch.Tensor([0]))


print(torch_conv_op(custom_input_T))

#Random image size test b/w pyTorch and Joey

weight_dev = torch_conv_op.weight.transpose(0,1).detach().numpy()


# convert the weights back to numpy so they can be used 
# in devito
bias_dev = torch_conv_op.bias.detach().numpy()

# pass the data to the PyTorch operator
torch_conv_res = torch_conv_op(input_data)
# Create the Joey Operator

layer1 = ml.Conv(kernel_size=(1, 3, 3),
                        input_size=(1, 1, image_size , image_size),
                        activation=None,
                        generate_code=True)
                        
# pass the data to the Joey operator
current_data = layer1.execute(input_data,bias_dev,weight_dev)
print("Same result:", np.allclose(torch_conv_res.detach().numpy(), current_data))
print("Max error across all indices :", np.max(abs(abs(torch_conv_res.detach().numpy()) - abs(current_data))))

print("PyTorch Weight sum:", np.sum(abs(torch_conv_res.detach().numpy())))
print("Joey Weight sum :", np.sum(abs(current_data)))
