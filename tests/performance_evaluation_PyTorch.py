from pickle import NONE
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch._C._autograd import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile

class Net(nn.Module):
        def __init__(self,image_size):
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




class lenet_PyTorch():
    image_size = 0
    def __init__(self,image_size, epochs,batch_size):
        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = Net(self.image_size)
        self.net.double()


    def train(self,trainloader):
        start_time = time.time()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        print("pyTorch start time", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        start_time = time.time()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        print("pyTorch start time", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))
        terminate = False
        if self.epochs is None:
                self.epochs = 1
                terminate = True
        start_time = time.time()
        for e in range(0,self.epochs): 
            for i, data in enumerate(trainloader, 0):
                images, labels = data
                optimizer.zero_grad()
                outputs = self.net(images.double())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if i==0:
                    with profile(activities=[ProfilerActivity.CPU],
                                    profile_memory=True, record_shapes=True) as prof:
                        outputs = self.net(images.double())
                                        
                    #print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))
                    if terminate:
                        break
            elapsed_time = time.time() - start_time
            print("batch :", self.batch_size, "itterations:", self.epochs, "pytorch: ", elapsed_time)

        pytorch_layers = [self.net.conv1, self.net.conv2, self.net.fc1, self.net.fc2, self.net.fc3]

        return pytorch_layers;

