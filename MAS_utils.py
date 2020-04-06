import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import os
import pdb
import math
import shutil
from torch.utils.data import DataLoader

def create_freeze_layers(model, num_layers = 2):
    """
    Returning the model with frozen layers for which require gradient argument to be false
    Inputs:
        model= 
        num_layers = No. of frozen layers
    Output:
        model, frozen layers
    """
    ## We want features no to be changed only the classifier learning according to tasks
    for param in model.xmodel.classifier.parameters():
        param.require_grad = True
    
    for param in model.xmodel.features.parameters():
        param.require_grad = False

    ## returning a n empty list of the num o layeers is zero
    if(num_layers==0):
        return []

    temporary =[]
    frozen_layers_data = []
    ## If we want 2 layers frozen then we have to make the requires_grad False for the num_layers number of conv2d in the feature module of the alexnet
    ## and we can do that by making all other's requires_grad True
    for key in model.xmodel.features._modules:
        if(type(model.xmodel.features._modules[key])==torch.nn.modules.conv.Conv2D):
            temp_list.append(key)

    no_non_frozen_layers = len(temp_list)-num_layers

    for no in range(0,no_non_frozen_layers):
        temp_key = temp_list[no]
        for param in model.xmodel.features[int(temp_key)].parameters():
            param.requires_grad = True

        name_1 = "features." + temp_key + ".weight"
        name_2 = "features." + temp_key + ".bias"

        frozen_layers_data.append(name1)
        frozen_layers_data.append(name2)
    
    return model,frozen_layers_data

