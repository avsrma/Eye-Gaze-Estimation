#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:51:11 2019

@author: avneesh
"""

import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, embedding_size = 1024):
        super(Encoder, self).__init__()
        
        self.first_conv = nn.Conv2d(1, 3, 1, padding = (1,1)) #create a new conv layer
        
        self.vgg = torchvision.models.vgg19(pretrained=True)  # pretrained ImageNet ResNet-101        
        self.vgg.classifier = nn.Linear(in_features=25088, out_features=1024)

        # add another fully connected layer
        self.embed = nn.Linear(in_features=1024, out_features=embedding_size)
        
        # dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
        # activation layers
        self.prelu = nn.PReLU()

#        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.first_conv(images)
        out = self.dropout(self.prelu(self.vgg(out)))
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.vgg.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.vgg.features)[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune



#class Attention(nn.Module):
#    """
#    Attention Network.
#    """
#
#    def __init__(self, encoder_dim, decoder_dim, attention_dim):
#        """
#        :param encoder_dim: feature size of encoded images
#        :param decoder_dim: size of decoder's RNN
#        :param attention_dim: size of the attention network
#        """
#        super(Attention, self).__init__()
#        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
#        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
#        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
#        self.relu = nn.ReLU()
#        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
#
#    def forward(self, encoder_out, decoder_hidden):
#        """
#        Forward propagation.
#        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
#        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
#        :return: attention weighted encoding, weights
#        """
#        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
#        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
#        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
#        alpha = self.softmax(att)  # (batch_size, num_pixels)
#        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
#
#        return attention_weighted_encoding, alpha

class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, poses_size, num_layers=1):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.poses_size = poses_size

        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=embedding_size, hidden_size=hidden_size)
    
        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.poses_size)

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.poses_size, embedding_dim=self.embedding_size)
    
        # activations
        #self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, features, poses):
        
        #input = [batch size]
        
        # batch size
        batch_size = features.size(0)
        
        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()
    
        # define the output tensor placeholder
        outputs = torch.empty((batch_size, poses.size(-1), self.poses_size)).cuda()

         # embed the poses
        poses_embed = self.embedding(poses)
        
        
        
  # pass the caption word by word
        for t in range(poses.size(1)):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
                
            # for the 2nd+ time step, using teacher forcer
            else:
                hidden_state, cell_state = self.lstm_cell(poses_embed[:, t, :], (hidden_state, cell_state))
            
            # output of the attention mechanism
            out = self.fc_out(hidden_state)
            
            # build the output tensor
            outputs[:, t, :] = out
    
        return outputs
    
    
    