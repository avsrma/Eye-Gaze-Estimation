#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:14:13 2019

@author: avneesh
"""

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
#        inception_firstlayer = models.inception_v3(pretrained = True) #load just the first conv layer
        inception = models.inception_v3(pretrained = True, aux_logits=False) #load upto the classification layers except first conv layer
        modules = list(inception.children())[0:16]  # exclude the first conv layer.
        
        
#        w1 = inception_firstlayer.state_dict()['weight'][:,0,:,:]
#        w2 = inception_firstlayer.state_dict()['weight'][:,1,:,:]
#        w3 = inception_firstlayer.state_dict()['weight'][:,2,:,:]
#        w4 = w1+w2+w3 # add the three weigths of the channels
#        w4 = w4.unsqueeze(1)# make it 4 dimensional
        
        first_conv = nn.Conv2d(1, 3, kernel_size=(1,1), padding = (1,1)) #create a new conv layer
#        print(len(first_conv.weight))
#        first_conv.weight = torch.nn.Parameter(w4, requires_grad=True) #initialize  the conv layer's weigths with w4
#        first_conv.bias = torch.nn.Parameter(inception_firstlayer.state_dict()['bias'], requires_grad=True) #initialize  the conv layer's weigths with vgg's first conv bias
    
    
        self.first_convlayer = first_conv #the first layer is 1 channel (Grayscale) conv layer
        self.inception = nn.Sequential(*modules)

        self.fc1 = nn.Linear(20480, 1000)
        self.fc2 = nn.Linear(1002, 2)
        
    def forward(self, x, y):
        x=self.first_convlayer(x)
        x=self.inception(x)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
        x = torch.cat([x, y], dim=1)
        x = self.fc2(x)

        return x

m = Model()
#print(m)