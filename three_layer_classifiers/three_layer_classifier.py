"""
This file will run any correlation cut and any channel type you want, and takes these in as command line arguments!

Because this does everything we want, I am going to move the other versions into the old/ duplicate directory since they are no longer needed.

If you want to add a version to the run then it will also take the version as a command line argument and it will be appended to the model name and everything saved which uses the model name. I will only leave one submission script in this directory, but you can find (and use by moving into this directory) other submission scripts so this can be run in parallel.
"""

# coding: utf-8

# In[26]:

######################################### Import statements ###########################################

import sys
sys.path.append("..")   # this allows us to import from sibling directory

from code_to_import.dataset_preprocessing import ProductionModeDataset

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

from matplotlib import pyplot as plt
from IPython import display


print(torch.cuda.is_available())


# In[28]:
################# Load Command Line Arguments ########################################################
parser = argparse.ArgumentParser()
parser.add_argument("--channel", type=str, default="ee", help="The channel type (ee, emu or mumu). Default is ee")
parser.add_argument("--cut",type=float, default=-1.0, help="The correlation cut being used. Default is -1, and negative cuts are not applied")
parser.add_argument("--version",type=int, default=-1, help="Version number of this type of run. Default is -1 which results in no number applied")
args = parser.parse_args()
print(args)


######################## THIS CLASS INCLUDES ALL THE VARIABLES YOU WANT TO CONFIGURE #################################
#######################################################################################################################

class opt():   # Class used for optimizers in the future. Defines all variables and stuff needed.
#     save_weights = True  # Tells whether to save weights... currently not used
    
    load_model = False  # set true if you want to load and continue training a previous model
    
    draw = False # set to false when running on slurm
    
    n_epochs = 40000   # an epoch is the number of times it works through the entire training set.
                       # This sets the total number of epochs which will be run
    
    batch_size = 5000   # the training set is broken up into batches, 
                        # this sets the size of each batch
    
    lr = 0.001   # learning rate (how much to change based on error)
    b1 = 0.9   # Used for Adam. Exponential decay rate for the first moment
    b2 = 0.999   # Used for Adam. Exponential decay rate for the second moment estimates (gradient squared)
        
    correlation_cut = args.cut   # this is the correlation cut... If negative then no cut is applied
    
    # the root_path leads to the folder with the root files being used for data
    root_path = "/depot-new/cms/top/mcnama20/TopSpinCorr-Run2-Entanglement/CMSSW_10_2_22/src/TopAnalysis/Configuration/analysis/diLeptonic/three_files/Nominal"

    file = root_path + "/" + args.channel + "_modified_root_1.root"   # this is the data root file loaded into the dataloader
    
    model_name = "threeLayerModel_" + args.channel + "_corrCut_" + str(correlation_cut) # this is the model name. 
                                                                          # Change it when running a new model
    # add version information if included
    if args.version > 0:
        model_name += str(args.version)
    
    # load data object so we can access validation and training data    
    if correlation_cut > 0:
        data = ProductionModeDataset(file, correlation_cut=correlation_cut)
    else:
        data = ProductionModeDataset(file)
        

print("The model in this run is " + opt.model_name)   # this will make slurm output easier to identify


cuda = True if torch.cuda.is_available() else False


############################## Configure data loader depending on if there is a correlation cut ###########################
if opt.correlation_cut > 0:
    os.makedirs("../data/three_layers/", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        opt.data,
        batch_size=opt.batch_size, drop_last=True,
        shuffle=True,
    )
else:
    os.makedirs("../data/three_layers/", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        opt.data,
        batch_size=opt.batch_size, drop_last=True,
        shuffle=True,
    )
print('done')

data = iter(dataloader)
x = data.next()
input_size = x.shape[1] -3

###################### Now save training and validation sets for future analysis ###################################
np.save("../data/three_layers/train_val_datasets/training_dataset_" + opt.model_name + ".npy", opt.data.get_training_data())
np.save("../data/three_layers/train_val_datasets/validation_dataset_" + opt.model_name + ".npy", opt.data.get_eval_data())


# In[29]:
########################################## CLASSIFIER CLASS #############################################################
#########################################################################################################################
class Classifier(nn.Module):
    """
    classifier layers
    """
    def __init__(self):
        super(Classifier, self).__init__()   # Just uses the module constructor with name Discriminator 

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),   # first layer
            nn.LeakyReLU(0.2, inplace=True),   # apply leaky relu to layer
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 3),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        """
        applies model to input and attempts to classify
        """
        output = self.model(input)   # Classifies the input (at location) as gg (0) qqbar (1) or other (2)
        return output


# ******* OUT OF CLASSES NOW ************

############### Initialize classifier and load a model if needed ##########################
classifier = Classifier()
if opt.load_model:
    classifier.load_state_dict(torch.load("../models/three_layers/" + opt.model_name + ".pt")) #load module with same name
    classifier.train()  # set the model up for training
if cuda:
    classifier.cuda()

    



# In[32]:

####################################### load data for evaluation of model (not training set) and separate weights and target
validation_data = opt.data.get_eval_data()

w_val = validation_data[:,input_size + 1]
w_val = Variable(torch.from_numpy(w_val).type(torch.FloatTensor))
target_val = validation_data[:,input_size]
target_val = Variable(torch.from_numpy(target_val).type(torch.LongTensor))
y_val = np.transpose(validation_data)
y_val = np.delete(y_val, [input_size, input_size + 1, input_size + 2], 0)
y_val = np.transpose(y_val)
val_data = Variable(torch.from_numpy(y_val).type(torch.Tensor))


# In[33]:

################### Initialize (and load if needed) optimizer and loss function ################################
################################################################################################################
optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)) # set up optimizer
if opt.load_model:
    optimizer.load_state_dict(torch.load("../models/three_layers/optimizer_" + opt.model_name + ".pt"))

criterion = nn.CrossEntropyLoss(reduction='none')# this is the loss function. reduction='none' makes it return a value for each input (reduce=False was deprecated)

##################################################################################################
################ initialize stuff before training ################################################

small_loss = 1e20   # This is the initail loss under which we overwrite the model.
                    # initialize with a large loss so everything is smaller than it

# initialize loss arrays
loss_val_array = np.array(())
loss_array = np.array(())
lva = 0  # this is the length of the loaded array
la = 0  # this is the length of the other loaded array

# load arrays and reset small_loss if loading model:
if opt.load_model:
    loss_val_array = np.load("/depot/cms/top/zhou907/ee_loss_val_array_1.npy")
    lva = len(loss_val_array)

    loss_array = np.load("/depot/cms/top/zhou907/ee_loss_array_1.npy")
    la = len(loss_array)
    
    small_loss = np.min(loss_val_array)
    

batches_done = 0   # Counter for batches


###############################################################################################################
# ******************************** START TRAINING LOOP *******************************************************
###############################################################################################################
for epoch in range(opt.n_epochs):   # Loop through all epochs
    for i, x in enumerate(dataloader): # x is a batch and there are i batches in the epoch
        
        #-----------------------------
        # Configure input
        #----------------------------
        variable_len = len(x[0])
        weight = x[:, variable_len-2]
        weight = Variable(weight.type(torch.FloatTensor))
        target = x[:, variable_len-3]
        target = Variable(target.type(torch.LongTensor))
        x = np.transpose(x)
        x = np.delete(x, [variable_len-3, variable_len-2, variable_len-1], 0)
        x = np.transpose(x)
        batch = Variable(x.type(torch.Tensor))   # Variable is a wrapper for the Tensor x was just made into

        # ---------------------
        #  Train Classifier
        # ---------------------

        optimizer.zero_grad()   # Make gradients zero so they don't accumulate
        
        output = classifier(batch)   # apply nn to input   

        # Calculate loss 
        loss_l = criterion(output, target) # loss_l is a vector with the loss for each event in the batch
        loss = torch.dot(weight,loss_l)   # we take the dot product with the weights to calculate the final loss
        loss.backward()   # Do back propagation 
        optimizer.step()   # Update parameters based on gradients for individuals
        batches_done += 1  # increase the number of batches in the counter
        
    
    #-----------------------------------------
    # Save and Draw Stuff  (if drawing is on)
    #-----------------------------------------
    if epoch % 10 == 0:
        print(loss.detach().numpy())
        loss_array = np.append(loss_array, loss.detach().numpy()) # append the training loss to the loss array
        
        
        out = classifier(val_data)   # run classifier on validation data to see how good it is
        loss_val = torch.dot(w_val, criterion(out, target_val))   # calculate the validation loss
        loss_val_array = np.append(loss_val_array, loss_val.detach().numpy()) # append the validation loss to its array
        
        if small_loss > loss_val:   # compare to see if the loss has decreased
            small_loss = loss_val   # if the network has improved replace the best loss with this one
            torch.save(classifier.state_dict(), "../models/three_layers/" + opt.model_name + ".pt")   # save the new (and better) model
            torch.save(optimizer.state_dict(), "../models/three_layers/optimizer_" + opt.model_name + ".pt") # save the optimizer state
            
        if epoch % 50 == 0:
            # save the loss arrays
            np.save("../data/three_layers/" + opt.model_name +  "_loss_val_array.npy",loss_val_array)
            np.save("../data/three_layers/" + opt.model_name + "_loss_train_array.npy",loss_array)
            if opt.draw:
                #----------------------------------
                # Draw training and validation loss
                #-------------------------------------
                display.clear_output(True)
                figure = plt.figure()
                ax = figure.add_subplot(111)
                #ax.set_yscale("log")
                ax.plot(np.array(list(range(int((epoch)/10)+lva+1))), loss_array, label="training loss")
                ax.plot(np.array(list(range(int((epoch)/10)+la+1))), loss_val_array, label = "evaluation loss")
                plt.legend()
                plt.show()



