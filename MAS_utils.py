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

from optimizers import *

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

        ## this list will contatin the name of the layers for which requires_grad is made true
        frozen_layers_data.append(name1)
        frozen_layers_data.append(name2)
    
    return model,frozen_layers_data

def initialising_omega(model,use_gpu,frozen_layers=[]):
    """
    Function:

        Inputs:

        Outputs:

    """
    device = torch.device("cuda:0" if use_gpu else "cpu")

    ## A dictionary to store the initialised omega values with the key and the values being the weights of the layer and importance value
    reg_params = {}

    ##Iterating over the the model in which name will give the name of the layer and param is giving the weights stored inside the correseponding layer
    ## For the frozen layers the omega will be calculated and reset every time new task is being trained
    for name, param in model.xmodel.named_parameters():
        if not name in frozen_layers:

            print ("Initialsing omega values for layer", name)
            omega = torch.zeros(param.size())
            omega = omega.to(device)

            init_val = param.data.clone()
            param_dict = {}

            ## init_val is just a variable which stores the weights of the corresponding layer
            #for first task, omega is initialised to zero
            param_dict["omega"]= omega
            param_dict["init_val"] = init_val ## storing the initial value

            reg_params[param] = param_dict ######

    model.params = reg_params

    return model

def check_checkpoints(storepath):
    if not os.path.exists(storepath):
        return ["",False]
    
    #directory exists but there is no checkpoint file
	onlyfiles = [f for f in os.listdir(store_path) if os.path.isfile(os.path.join(store_path, f))]
	max_train = -1
	flag = False

	#Check the latest epoch file that was created
	for file in onlyfiles:
		if(file.endswith('pth.tr')):
			flag = True
			test_epoch = file[0]
			if(test_epoch > max_train): 
				max_epoch = test_epoch
				checkpoint_file = file
	
	#no checkpoint exists in the directory so return an empty string
	if (flag == False): 
		checkpoint_file = ""

    return [checkpoint_file,flag]
    
def create_task_dir( num_classes, store_path):
    os.makedir(store_path)
    file = os.path.join(store_path,"classes.txt")
    with open(file_path,'w') as file1:
        input_to_textfile = str(num_classes)
        file1.write(input_to_text_file)
        file1.close()

    return