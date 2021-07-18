
# coding: utf-8

# In[1]:

from dataset_preprocessing import ProductionModeDataset

import torch
from sklearn.inspection import permutation_importance
import numpy as np
import uproot
import torch.nn as nn
from matplotlib import pyplot as plt

from Classifier_module import Classifier
from torch.autograd import Variable


# In[2]:

################# THIS IS WHERE YOU CHOOSE WHAT TO LOAD ################################
path_to_model = "./models/bad/twoLayerModel_bad_emu0"

event_type = "emu"  # could be ee, emu, mumu
root_path = "/depot-new/cms/top/mcnama20/TopSpinCorr-Run2-Entanglement/CMSSW_10_2_22/src/TopAnalysis/Configuration/analysis/diLeptonic/three_files/Nominal"

file = root_path + "/" + event_type + "_modified_root_1.root"


# In[3]:

# model = 0
model = Classifier()
model.load_state_dict(torch.load(path_to_model))
model.eval()


# In[4]:




data_object = ProductionModeDataset(file)
x = data_object.get_eval_data()


# In[5]:

weight = x[:,81]
target = x[:,80]
target = Variable(torch.from_numpy(target).type(torch.LongTensor))
y = np.transpose(x)
y = np.delete(y, [80, 81, 82], 0)
y = np.transpose(y)
input = Variable(torch.from_numpy(y).type(torch.Tensor))


# In[6]:

model(input)[0,:]


# In[7]:

target[0]


# In[8]:

from skorch import NeuralNetClassifier
from sklearn.inspection import permutation_importance


# In[9]:

nn = NeuralNetClassifier(model, max_epochs=0, lr=0.00, batch_size=5000)
nn.initialize()


# In[10]:

events = uproot.open(file)
key = events.keys()[0]
input_variables = events[key].keys()


# In[11]:

r = permutation_importance(nn, input, target,                            n_repeats=30,                            random_state=0)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{input_variables[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")

