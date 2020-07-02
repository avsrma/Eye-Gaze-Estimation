# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:15:05 2019

@author: iamav
"""

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        vgg_firstlayer=models.vgg16(pretrained = True).features[0] #load just the first conv layer
        vgg_1=models.vgg16(pretrained = True).features[1:10] #load upto the classification layers except first conv layer
        vgg_2=models.vgg16(pretrained = True).features[10:17]
        vgg_3=models.vgg16(pretrained = True).features[17:24]
        vgg_4=models.vgg16(pretrained = True).features[24:30]
        
        w1=vgg_firstlayer.state_dict()['weight'][:,0,:,:]
        w2=vgg_firstlayer.state_dict()['weight'][:,1,:,:]
        w3=vgg_firstlayer.state_dict()['weight'][:,2,:,:]
        w4=w1+w2+w3 # add the three weigths of the channels
        w4=w4.unsqueeze(1)# make it 4 dimensional
    
        first_conv=nn.Conv2d(1, 64, 3, padding = (1,1)) #create a new conv layer
        first_conv.weight=torch.nn.Parameter(w4, requires_grad=True) #initialize  the conv layer's weigths with w4
        first_conv.bias=torch.nn.Parameter(vgg_firstlayer.state_dict()['bias'], requires_grad=True) #initialize  the conv layer's weigths with vgg's first conv bias
    
    
        self.first_convlayer=first_conv #the first layer is 1 channel (Grayscale) conv layer
        self.vgg_1 =nn.Sequential(vgg_1)
        self.dialated_conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), dilation=(2,2))
        self.vgg_2 =nn.Sequential(vgg_2)
        self.dialated_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), dilation=(4,4))
        self.vgg_3 =nn.Sequential(vgg_3)
        self.dialated_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), dilation=(6,6))
        self.vgg_4 =nn.Sequential(vgg_4)
        self.dilatedpool = nn.MaxPool2d(kernel_size=2, stride=2, dilation=(2,2))
        
        self.fc1 = nn.Linear(3072, 1000)
        self.fc2 = nn.Linear(1002, 2)

    def forward(self, x, y):
        x=self.first_convlayer(x)
        x=self.vgg_1(x)        
        x=self.vgg_2(x)
        x=self.vgg_3(x)
        x=self.vgg_4(x)
        
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
        x = torch.cat([x, y], dim=1)
        x = self.fc2(x)

        return x
    
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)

class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()

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
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
        x = torch.cat([x, y], dim=1)
        x = self.fc2(x)
        return x

class Ensemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(4, 2)
        
    def forward(self, x, y):
        x1 = self.modelA(x,y)
        x2 = self.modelB(x,y)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x

modelA = ModelA()
modelB = ModelB()


model = Ensemble(modelA, modelB)
print(model.parameters)

