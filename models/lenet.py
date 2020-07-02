# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(3600, 500)  
        self.fc2 = nn.Linear(502, 2)

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        self.apply(initialize_weights)

    def forward(self, x, y):
        #print("input shape: " + str(x.shape))
        #print("conv1 " + str(self.conv1.parameters))
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
#        print("pool1"+ str(x))
        #print("conv2 " + str(self.conv2.parameters))
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
#        print("pool2"+ str(x))
#        print("x before r & fc ", x.shape)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
#        print("flattened " + str(x.shape))
#        print("y shape ", y.shape)
        x = torch.cat([x, y], dim=1)
#        print("cat x", x.shape)
#        print("cat y", y.shape)
        x = self.fc2(x)
        
#        print("output "+ str(x.shape))
        return x

"""
#poses shape:  torch.Size([64, 2])
#im shape:  torch.Size([64, 1, 36, 60])
#x before r & fc  torch.Size([64, 50, 6, 12])
#flattened torch.Size([64, 500])
#y shape  torch.Size([64, 2])
#cat x torch.Size([64, 502])
#cat y torch.Size([64, 2])
#output torch.Size([64, 2])
"""
