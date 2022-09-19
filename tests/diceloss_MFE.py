from diceloss import DiceLoss
import torch
import numpy as np
from torch.autograd import grad
torch.manual_seed(0)

input = torch.rand((5,5), requires_grad=True)

output = torch.rand((5,5),requires_grad=True)

def dice_loss_grad(y, label):
        sum_squares = np.sum(y**2)+np.sum(label**2)
        num =  (label * sum_squares) - (2*y*np.sum(y*label))
        dem =   (sum_squares)**2 
        res = -2*(num/dem)
        return res

c = DiceLoss()
loss = c(input, output)
loss.backward(retain_graph=True)

result_torch = grad(outputs=loss, inputs=input, retain_graph=True)[0].detach().numpy()
loss_j = dice_loss_grad(input.detach().numpy(), output.detach().numpy())
print(result_torch)
print(loss_j)