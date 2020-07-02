# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 21:13:51 2019

@author: iamav
"""
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
        self.conv1 = nn.Conv2d(1, 256, kernel_size=11, padding=0)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=5)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=5)
        self.conv4 = nn.Conv2d(1024, 1024, kernel_size=5)
        self.conv5 = nn.Conv2d(1024, 1024, kernel_size=5)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 2)
        
        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
#        nn.init.normal_(self.conv3.weight, std=0.001)
        self.apply(initialize_weights)

    def forward(self,x,y):
        x = x
        print(x) ## Input shape is [64, 1, 36, 60] 
        x = F.relu(self.conv1(x)) ## Shape is [64, 256, 26, 50]
        print(x.shape)

        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        print(x.shape)
        
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        print(x.shape)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(self.conv3(x), kernel_size=2, stride=2)
        
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(self.conv4(x), kernel_size=2, stride=2)
        
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(self.conv5(x), kernel_size=2, stride=2)
        
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten
        x = torch.cat([x, y], dim=1)
        x = self.fc2(x)

        return x
