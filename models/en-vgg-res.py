# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:42:56 2019

@author: iamav
"""

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        vgg_firstlayer=models.vgg19(pretrained = True).features[0] #load just the first conv layer
        vgg=models.vgg19(pretrained = True).features[1:36] #load upto the classification layers except first conv layer

        w1=vgg_firstlayer.state_dict()['weight'][:,0,:,:]
        w2=vgg_firstlayer.state_dict()['weight'][:,1,:,:]
        w3=vgg_firstlayer.state_dict()['weight'][:,2,:,:]
        w4=w1+w2+w3 # add the three weigths of the channels
        w4=w4.unsqueeze(1)# make it 4 dimensional
    
        first_conv=nn.Conv2d(1, 64, 3, padding = (1,1)) #create a new conv layer
        first_conv.weight=torch.nn.Parameter(w4, requires_grad=True) #initialize  the conv layer's weigths with w4
        first_conv.bias=torch.nn.Parameter(vgg_firstlayer.state_dict()['bias'], requires_grad=True) #initialize  the conv layer's weigths with vgg's first conv bias
    
    
        self.first_convlayer=first_conv #the first layer is 1 channel (Grayscale) conv layer
        self.vgg =nn.Sequential(vgg)

        self.fc1 = nn.Linear(3072, 1000)
        self.fc2 = nn.Linear(1002, 2)

    def forward(self, x, y):
        x=self.first_convlayer(x)
        x=self.vgg(x)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
        x = torch.cat([x, y], dim=1)
        x = self.fc2(x)

        return x
    

class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        resnet_firstlayer = models.resnet50(pretrained = True).conv1 #load just the first conv layer
        resnet = models.resnet50(pretrained = True) #load upto the classification layers except first conv layer
        modules = list(resnet.children())[1:9]  # not include the first conv layer.
        
        
        w1 = resnet_firstlayer.state_dict()['weight'][:,0,:,:]
        w2 = resnet_firstlayer.state_dict()['weight'][:,1,:,:]
        w3 = resnet_firstlayer.state_dict()['weight'][:,2,:,:]
        w4 = w1+w2+w3 # add the three weigths of the channels
        w4 = w4.unsqueeze(1)# make it 4 dimensional
    
        first_conv = nn.Conv2d(1, 64, 3, padding = (1,1)) #create a new conv layer
        first_conv.weight = torch.nn.Parameter(w4, requires_grad=True) #initialize  the conv layer's weigths with w4
#        first_conv.bias = torch.nn.Parameter(resnet_firstlayer.state_dict()['bias'], requires_grad=True) #initialize  the conv layer's weigths with vgg's first conv bias
    
    
        self.first_convlayer = first_conv #the first layer is 1 channel (Grayscale) conv layer
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1002, 2)

    def forward(self, x, y):
        x=self.first_convlayer(x)
        x=self.resnet(x)
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

# Create models and load state_dicts    
modelA = ModelA()
modelB = ModelB()


model = Ensemble(modelA, modelB)
print(model.parameters)
