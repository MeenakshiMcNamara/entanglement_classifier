
# coding: utf-8

# In[2]:

import numpy as np
import os
import torch
import uproot
from torch.utils.data import Dataset


# In[3]:

class ProductionModeDataset(Dataset):
    """
    This class will load the minitree and create the dataset loaded into the nns
    
    0) load the minitree properly
    
    1) remove 0 and 2 type events so we have the same number as qqbar
    
    2) normalize all data
    
    3) ensure this is stored in a numpy array with the correct arrangement of data and a corresponding list of datatypes
    
    4) split this up into training and evaluating datasets
    
    5) provide methods with which these sets can be accessed
    
    Args:
        :param root (string): is root directory and file name of minitree
        :param split (boolean): tells whether to split into training and eval
        :param normalize (boolean): tells whether to normalize data
        :param remove (boolean): tells whether we should remove excess data for non-qqbar events or duplicate qqbar
        :param train (boolean): tells whether we are training or evaluating (probably not needed)
        :param correlation_cut(float): if positive, removes values correlated more than the correlation cut. Requires existing 
                                analysis of the given cut so that data can be loaded
        : param cut_version (int): if positive, loads the specific cut version (otherwise loads unnumbere OG version)
        : param include_qg (boolean): default true. If false then remove stuff with production mode 2
    """
    
    def __init__(self, root, split=True, normalize=True, remove=True, train=True, correlation_cut=-1.0, cut_version=-1, 
                 include_qg = True):
        # load a correlation cut if it exists:
        to_remove = np.array(())  # initialize the array of inputs to remove because of cut as empty
        if correlation_cut > 0:
            # load the inputs which must be removed due to cut if positive
            if cut_version > 0:
                to_remove = np.load("../analysis_code/results/inputs_to_remove_cut_" + str(correlation_cut)\
                                    + "v" + str(cut_version) + ".npy")
                print("loaded correlations... shape is " + str(to_remove.shape))
            else:
                to_remove = np.load("../analysis_code/results/inputs_to_remove_cut_" + str(correlation_cut) + ".npy")
                print("loaded correlations... shape is " + str(to_remove.shape))
        
        self.events = uproot.open(root)   # open the root file and load the events to be processed
        self.training = train    # set a variable which determines whether this is loaded for training. probably not needed
        
        key = self.events.keys()[0]   # get the key to make accessing events easier
        data_list = self.events[key].keys()   # get the names of the inputs
        
        # make a numpy array from events array
        self.events_array = np.array([self.events[key + "/" + k].array(library="np") for k in data_list])
        print(data_list)
    
        self.events_array = np.transpose(self.events_array)   # turn into columns (instead of rows) of data

        
        ################# remove any inputs in to_remove from the events array and the list of inputs ################
        if to_remove.shape[0] > 0:
            print(to_remove.shape)
            self.events_array = np.delete(self.events_array, to_remove, 1)
            to_remove = np.sort(to_remove)
            for i in range(len(to_remove)):
                to_remove[i] -= i
                data_list.pop(to_remove[i])
        ##############################################################################################################
        
        
        if remove:
            """
            Here is where we remove the excess data so that qqbar, gg, and gq/qg events are equally
            represented in the training and analysis.
            """
            # find index of production_mode:
            index = 0
            for i in range(len(data_list)):
                if data_list[i] == "production_mode":
                    index = i
                    break
                    
            # sort events array by production mode
            self.events_array = self.events_array[np.argsort(self.events_array[:, index])]           
            
            ############## find first and last index with production mode 1 #########################
            first = 0  # this will be the first index with qqbar
            last = 0   # this will be the last index with qqbar
            found_first = False   # a flag which allows us to stop looking for first once found
            
            for i in range(len(self.events_array[:,index])):
                if not found_first:
                    if self.events_array[i ,index] == 1:
                        first = i
                        found_first = True
                
                if self.events_array[i,index] == 2:
                    last = i - 1
                    break
            #######################################################################################
                    
            num_qqbar = last + 1 - first  # this is the total number of qqbar events
            print("num qqbar = " + str(num_qqbar))
            
            ####################### remove the extra gg and other #############################################
            max_len = len(self.events_array[:,0])
            
            if include_qg:
                """
                keep every production mode but remove excess events beyond qqbar amount
                """ 
                self.events_array = np.delete(self.events_array, list(range(num_qqbar, first)) + \
                                              list(range(max_len - (max_len - (last + 1) -num_qqbar), max_len)), 0)
                
            elif not include_qg:
                """
                Here we remove all quark-gluon events if we don't want them
                """
                self.events_array = np.delete(self.events_array, list(range(num_qqbar, first)) + \
                                             list(range(last, max_len)), 0)
                
            #################################################################################################
            
        if not remove:
            """
            Here is where we duplicate qqbar data so that qqbar and gg events are equally represented in the analysis
            """
            # find index of production_mode:
            np.random.shuffle(self.events_array)
            index = 0
            for i in range(len(data_list)):
                if data_list[i] == "production_mode":
                    index = i
                    break
                    
            # sort events array by production mode
            self.events_array = self.events_array[np.argsort(self.events_array[:, index])]           
            
            ############## find first and last index with production mode 1 #########################
            first = 0  # this will be the first index with qqbar
            last = 0   # this will be the last index with qqbar
            found_first = False   # a flag which allows us to stop looking for first once found
            
            for i in range(len(self.events_array[:,index])):
                if not found_first:
                    if self.events_array[i ,index] == 1:
                        first = i
                        found_first = True
                
                if self.events_array[i,index] == 2:
                    last = i - 1
                    break
            #######################################################################################
            num_qqbar = last + 1 - first  # this is the total number of qqbar events
            print("num qqbar = " + str(num_qqbar))
            
            num_gg = first - 1
            print("num gg is " + str(num_gg))
            
            max_len = len(self.events_array[:,0])
            #########################################################################
#             new_events = [self.events_array[first + np.mod(loop, num_qqbar),:] for loop in range(num_gg - num_qqbar)]
#             new_events = np.array(new_events)
#             print("there were this many new events " + str(len(new_events)))
#             self.events_array = np.vstack([self.events_array, new_events])
#             print("new total length is " + str(len(self.events_array[:,index])))
#             print(self.events_array)
            
            ####################### remove the extra gg and other #############################################
            
            
            if include_qg:
                """
                keep every production mode but remove excess events beyond qqbar amount
                """ 
                self.events_array = np.delete(self.events_array, list(range(num_qqbar, first)) + \
                                              list(range(max_len - (max_len - (last + 1) -num_qqbar), max_len)), 0)
                
            elif not include_qg:
                """
                Here we remove all quark-gluon events if we don't want them
                """
                self.events_array = np.delete(self.events_array, list(range(last + 1,max_len)),0)
                print("range is " + str(range(last + 1,max_len)))
                
            print("final total length is " + str(len(self.events_array)))
            
            #################################################################################################
            
  
        if normalize:
            """
            Here we normalize all the data by subtracting the minimum and dividing by the range (max-min). We do not normalize
            anything not going to be used as an input to the network (last three columns) except the weights which are only
            divided by the max to ensure we keep the signs.
            """
            # loop through all the inputs. the last three are not looped through because they are not input to the 
            # neural network. Hence, we subtract 3 from the length.
            for i in range(len(self.events_array[0,:])-3):
                ori_a = self.events_array[:,i] # original input column 

                min_a = np.min(self.events_array[:,i]) # min
                max_a = np.max(self.events_array[:,i]) # max
                range_a = max_a - min_a #range

                if range_a > 0:
                    self.events_array[:, i] = (ori_a - min_a) / (range_a) # replace the og array with the normalized array
                
            #################### normalize the weights here: ######################################################
            n = len(self.events_array[0,:])-2   # this is the index of the weights for reco level inputs
            self.events_array[:,n] /= np.max(self.events_array[:,n])   # replace weights with normalized weights
            #######################################################################################################
        
        
               
        if split:
            """
            Here we split the events array into a validation dataset and a training dataset. 80% will be training and 
            20% will be validation.
            """
            # shuffle so no longer sorted by production modebefore splitting
            np.random.shuffle(self.events_array)
            
            train_size = int(len(self.events_array[:,0])*80/100)  # calculate the size of the training array
            
            # split into the two arrays
            self.training_array, self.eval_array = np.split(self.events_array, [train_size])
            print("training before upsampling" + str(self.training_array.shape))
            print("evaluating " + str(self.eval_array.shape))
            
        else:
            """
            If not splitting, just set both the training_array and eval_array to be the entire dataset
            """
            self.training_array = self.events_array
            self.eval_array = self.events_array
            
        ################################## Upsampling ####################################################
            
        if remove == False:
            """
            Sort, copy qqbar to upsample, and then shuffle to random
            """
            # find index of production_mode:
#             np.random.shuffle(self.training_array)
            index = 0
            for i in range(len(data_list)):
                if data_list[i] == "production_mode":
                    index = i
                    break
                    
            # sort events array by production mode
            self.training_array = self.training_array[np.argsort(self.training_array[:, index])]           
            
            ############## find first and last index with production mode 1 #########################
            first = 0  # this will be the first index with qqbar
            last = len(self.training_array[:,index])-1   # this will be the last index with qqbar
            found_first = False   # a flag which allows us to stop looking for first once found
            
            for i in range(len(self.training_array[:,index])):
                if not found_first:
                    if self.training_array[i ,index] == 1:
                        first = i
                        found_first = True
                        break
                
            #######################################################################################
            num_qqbar = last + 1 - first  # this is the total number of qqbar events
            print("num qqbar = " + str(num_qqbar))
            
            num_gg = first - 1
            print("num gg is " + str(num_gg))
            
            max_len = len(self.training_array[:,0])

            
            new_events = [self.training_array[first + np.mod(loop, num_qqbar),:] for loop in range(num_gg - num_qqbar)]
            new_events = np.array(new_events)
            print("there were this many new events " + str(len(new_events)))
            self.training_array = np.vstack([self.training_array, new_events])
            print("new total length is " + str(len(self.training_array[:,index])))
            print(self.training_array)
            
            np.random.shuffle(self.training_array)
            
                
    def __getitem__(self, index):
        """
        Return the training array (should never be eval... might delete stuff) in format needed for dataloader 
        """
        if self.training:
            return self.training_array[index]
        return self.eval_array[index]
    
    def __len__(self):
        """
        Return the length of the training array (once again, might want to delete extra code for eval)
        """
        if self.training:
            return len(self.training_array[:,0])
        return len(self.eval_array[:,0])
    
    def get_eval_data(self):
        """ Return the entire validation dataset """
        return self.eval_array
    
    def get_training_data(self):
        """ Return the entire training dataset """
        return self.training_array