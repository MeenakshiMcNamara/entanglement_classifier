
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
    
    Args:
        :param root (string): is root directory and file name of minitree
        :param split (boolean): tells whether to split into training and eval
        :param normalize (boolean): tells whether to normalize data
        :param remove (boolean): tells whether we should remove excess data for non-qqbar events or duplicate qqbar
        :param train (boolean): tells whether we are training or evaluating
    """
    
    def __init__(self, root, split=True, normalize=True, remove=True, train=True, correlation_cut=-1.0):
        # load a correlation cut if it exists:
        to_remove = np.array(())
        if correlation_cut > 0:
            to_remove = np.load("../analysis_code/results/inputs_to_remove_cut_" + str(correlation_cut) + ".npy")
            print("loaded correlations... shape is " + str(to_remove.shape))
        
        self.events = uproot.open(root)
        self.training = train
        #self.events = self.events['Events']
        
        #data_list = data = uproot.open(depo_dir + root_f, key = "ttBar_treeVariables_step8;3")
        
        
        # TODO: make less complex when not needed later
#         data_list = [x in self.events["ttBar_treeVariables_step8;3"].keys() if ((("phi" in x) or ("eta" in x) \
#                                                                              or ("pt" in x) or ("production" in x)) and \
#                                                                              ("gen" not in x))]
#         print(str(data_list))
        key = self.events.keys()[0]
        data_list = self.events[key].keys()
    
        self.events_array = np.array([self.events[key + "/" + k].array(library="np") for k in data_list])
        print(data_list)
    
        self.events_array = np.transpose(self.events_array)   # turn to columns of data

        if to_remove.shape[0] > 0:
            self.events_array = np.delete(self.events_array, to_remove, 1)
            to_remove = np.sort(to_remove)
            for i in range(len(to_remove)):
                to_remove[i] -= i
                data_list.pop(to_remove[i])

        
        if remove:
            """
            Here is where we remove the excess data
            """
            # find index of production_mode:
            index = 0
            for i in range(len(data_list)):
                if data_list[i] == "production_mode":
                    index = i
                    break
                    
            self.events_array = self.events_array[np.argsort(self.events_array[:, index])]            
            
            # find first and last index with production mode 1
            first = 0
            last = 0
            found_first = False
            
            for i in range(len(self.events_array[:,index])):
                if not found_first:
                    if self.events_array[i ,index] == 1:
                        first = i
                        found_first = True
                
                if self.events_array[i,index] == 2:
                    last = i - 1
                    break
                    
            num_qqbar = last + 1 - first
            print("num qqbar = " + str(num_qqbar))
            # remove the extra gg and other
            max_len = len(self.events_array[:,0])
            self.events_array = np.delete(self.events_array, list(range(num_qqbar, first)) + \
                                          list(range(max_len - (max_len - (last + 1) -num_qqbar), max_len)), 0)
            
        # normalize here
            #print(len(self.events_array[:,0]))
            #columns = self.events_array[1]
            
            #min_a = []
            #max_a = []
            #range_a = []
            #update_a = []
        if normalize:
            for i in range(len(self.events_array[0,:])-3):#the last three is subtracted, thust not there any more
                ori_a = self.events_array[:,i]#original number
                #print('ori_a =',ori_a)
                min_a = np.min(self.events_array[:,i])#min
                max_a = np.max(self.events_array[:,i])#max
                range_a = max_a - min_a #range
                #print('range =',range_a)
                self.events_array[:, i] = (ori_a - min_a) / (range_a) #normalized list. i dont know what to do with it.
#                 print('normalized =', np.max(self.events_array[:,i]))
                
            # normalize the weights here:
            n = len(self.events_array[0,:])-2
            self.events_array[:,n] /= np.max(self.events_array[:,n])
        
        
        # split here        
        if split:
            # shuffle so no longer sorted before splitting
            np.random.shuffle(self.events_array)
            
            train_size = int(len(self.events_array[:,0])*81/100)
#             eval_size = len(self.events_array[:,0]) - train_size
            
            self.train_array, self.eval_array = np.split(self.events_array, [train_size])
            print("training " + str(self.train_array.shape))
            print("evaluating " + str(self.eval_array.shape))
            
        else:
            self.train_array = self.events_array
            self.eval_array = self.events_array
                
    def __getitem__(self, index):
        if self.training:
            return self.train_array[index]
        return self.eval_array[index]
    
    def __len__(self):
        if self.training:
            return len(self.train_array[:,0])
        return len(self.eval_array[:,0])
    
    def get_eval_data(self):
        return self.eval_array
    
    def get_training_data(self):
        return self.train_array