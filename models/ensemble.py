# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:42:56 2019

@author: iamav
"""

### Ensemble VGG19 and LeNet


import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        vgg_firstlayer=models.vgg19(pretrained = True).features[0] #load just the first conv layer
        vgg=models.vgg19(pretrained = True).features[1:37] #load upto the classification layers except first conv layer

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
        
        self.avgpool = models.vgg19(pretrained=True).avgpool

        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1026, 256)
        self.fc3 = nn.Linear(256, 2)
        self.fc4 = nn.Linear(2,2)

    def forward(self, x, y):
        x=self.first_convlayer(x)
        x=self.vgg(x)
        x = self.avgpool(x)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten        
        x = torch.cat([x, y], dim=1)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

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
        #x = F.relu(x.view(x.size(0), -1)) #flatten
        x = torch.cat((x1, x2), dim=1)
#        print("catx12", x.shape)                ### 128x4
        x = self.classifier(F.relu(x))
#        print("cls(relu x )", x.shape)
#        x = torch.cat([x,y], dim=1)
#        print("final cat xy", x.shape)
        return x

# Create models and load state_dicts    
modelA = ModelA()
modelB = ModelB()

model = Ensemble(modelA, modelB)
print(model.parameters)
#x1, x2 = torch.randn(1, 1000), torch.randn(1, 500)
#output = model(x1, x2)
















#writer = SummaryWriter()
##writer.add_graph(model, verbose=True)
#writer.close()
#
#x = torch.zeros(1, 1, 36, 60, dtype=torch.float, requires_grad=False)
#y = torch.zeros(64,2)
#out = model(x,y)
#make_dot(out)  # plot graph of variable, not of a nn.Module




