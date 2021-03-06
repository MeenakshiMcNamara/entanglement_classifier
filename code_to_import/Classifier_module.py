"""
This file contains the classes which will be imported so that we can load modules for analysis. We can either continue to add additional
classes as more network archetectures are produced, or we can add an input to the initialization of a single Classifier class (e.g. __init__(self, num_layers) and then construct the number of layers based on that
"""
# coding: utf-8

# In[ ]:

import argparse
import os
import numpy as np
import math
import sys
#import torchvision.transforms as transforms
#from torchvision.utils import save_image

from torch.utils.data import DataLoader
#from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F   # NOTE: I don't think this is used
import torch.autograd as autograd
import torch
from dgl.nn import GraphConv
import dgl



# In[ ]:

class opt():   # Class used for optimizers in the future. Defines all variables and stuff needed.
    save_weights = False
    n_epochs = 30000   # an epoch is the number of times it works through the entire training set
    batch_size = 5000   # the training set is broken up into batches, 
                        # and the average loss is used from a given batch for back propagation
    lr = 0.001   # learning rate (how much to change based on error)
    b1 = 0.9   # Used for Adam. Exponential decay rate for the first moment
    b2 = 0.999   # Used for Adam. Exponential decay rate for the second moment estimates (gradient squared)
    input_size = 80
    data_file = "/data"
    config_dir = "."
    save_location = config_dir + data_file
    root_path = "/depot/darkmatter/data/jupyterhub/Physics_Undergrads/Steve/things"

    file = root_path + "/all_1.root"

    #n_cpu = 2    not used rn

# os.makedirs(config_dir, exit_ok=True)

cuda = True if torch.cuda.is_available() else False


# In[ ]:

class Classifier(nn.Module):
    """
    classifier layers
    """
    def __init__(self):
        super(Classifier, self).__init__()   # Just uses the module constructor with name Discriminator 

        self.model = nn.Sequential(
            nn.Linear(opt.input_size, 256),   # first layer
            nn.LeakyReLU(0.2, inplace=True),   # apply leaky relu to layer
            nn.Linear(256, 3),   # last layer
            nn.LeakyReLU(0.2, inplace=True)   # apply leaky relu
        )

    def forward(self, input):
        """
        applies model to input and attempts to classify
        """
        output = self.model(input)   # Classifies the input (at location) as gg (0) qqbar (1) or other (2)
        return output
    
class Two_Layer_Classifier(nn.Module):
    """
    classifier layers
    """
    def __init__(self, input_size=opt.input_size, number_of_classes=3, batch_norm=True, drop=True, drop_val=0.0):
        super(Two_Layer_Classifier, self).__init__()   # Just uses the module constructor with name Discriminator 
        
        if batch_norm and not drop:
            self.model = nn.Sequential(
                nn.Linear(input_size, 256),   # first layer
                nn.BatchNorm1d(256),   # batch normalization
                nn.LeakyReLU(0.2, inplace=True),   # apply leaky relu to layer
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),# batch normalization
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, number_of_classes),
                nn.BatchNorm1d(number_of_classes),  # batch normalization
                nn.LeakyReLU(0.2, inplace=True)
            )
            
        elif not batch_norm:
            self.model = nn.Sequential(
                nn.Linear(input_size, 256),   # first layer
                nn.LeakyReLU(0.2, inplace=True),   # apply leaky relu to layer
                nn.Linear(256, number_of_classes),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
        elif drop:   # assuming batch_norm always true with drop
            self.model = nn.Sequential(
                nn.Linear(input_size, 256),   # first layer
                nn.BatchNorm1d(256),   # batch normalization
                nn.Dropout(drop_val),   # add dropout
                nn.LeakyReLU(0.2, inplace=True),   # apply leaky relu to layer
                nn.Linear(256, number_of_classes),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, input):
        """
        applies model to input and attempts to classify
        """
        output = self.model(input)   # Classifies the input (at location) as gg (0) qqbar (1) or other (2)
        return output


    
class Three_Layer_Classifier(nn.Module):
    """
    classifier layers
    """
    def __init__(self, input_size=opt.input_size, number_of_classes=3, batch_norm=True, drop=True, drop_val=0.0):
        super(Three_Layer_Classifier, self).__init__()   # Just uses the module constructor with name Discriminator 
        
        if batch_norm and not drop:
            self.model = nn.Sequential(
                nn.Linear(input_size, 512),   # first layer
                nn.BatchNorm1d(512),   # batch normalization
                nn.LeakyReLU(0.2, inplace=True),   # apply leaky relu to layer
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),# batch normalization
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, number_of_classes),
                nn.BatchNorm1d(number_of_classes),  # batch normalization
                nn.LeakyReLU(0.2, inplace=True)
            )
            
        elif not batch_norm:
            self.model = nn.Sequential(
                nn.Linear(input_size, 512),   # first layer
                nn.LeakyReLU(0.2, inplace=True),   # apply leaky relu to layer
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, number_of_classes),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
        elif drop:   # assuming batch_norm always true with drop
            self.model = nn.Sequential(
                nn.Linear(input_size, 512),   # first layer
                nn.BatchNorm1d(512),   # batch normalization
                nn.Dropout(drop_val),   # add dropout
                nn.LeakyReLU(0.2, inplace=True),   # apply leaky relu to layer
                nn.Linear(512, 256),
                nn.Dropout(drop_val),   # add dropout
                nn.BatchNorm1d(256),# batch normalization
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, number_of_classes),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, input):
        """
        applies model to input and attempts to classify
        """
        output = self.model(input)   # Classifies the input (at location) as gg (0) qqbar (1) or other (2)
        return output

class Five_Layer_Classifier(nn.Module):
    """
    classifier layers
    """
    def __init__(self, input_size=opt.input_size, number_of_classes=3, batch_norm=True, drop=True, drop_val=0.0):
        super(Five_Layer_Classifier, self).__init__()   # Just uses the module constructor with name Discriminator 

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),   # first layer
            nn.BatchNorm1d(64),   # batch normalization
            nn.LeakyReLU(0.2, inplace=True),   # apply leaky relu to layer
            nn.Dropout(drop_val),   # add dropout
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),# batch normalization
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop_val),   # add dropout
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),# batch normalization
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop_val),   # add dropout
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),# batch normalization
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop_val),   # add dropout
            
            nn.Linear(64, number_of_classes),   # last layer
            nn.LeakyReLU(0.2, inplace=True))
        
        
        
    def forward(self, input):
        """
        applies model to input and attempts to classify
        """
        output = self.model(input)   # Classifies the input (at location) as gg (0) qqbar (1) or other (2)
        return output
    
    

class GCN4Layers(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GCN4Layers, self).__init__()
        self.conv1 = GraphConv(in_feats, 128)
        self.conv2 = GraphConv(128, 64)
        self.conv3 = GraphConv(64, 32)
        self.conv4 = GraphConv(32, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g,h)
        h = F.relu(h)
        h = self.conv4(g,h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')
