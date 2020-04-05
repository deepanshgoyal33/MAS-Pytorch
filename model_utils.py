import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import copy
import os
import shutil
import pickle

from model import *

def model_initialiser(no_of_classes,use_gpu):

    """
    We have to delete the classifiaction head of the model coming from the previous task so we take the model   
    and detach its classification head and put a new one and initialise another model with new classification head
    but the features are shared.
    Inputs-

    Output-

    """
    pre_model= SharedModel()

    ## New classifiaction head
    in_features = pre_model.model.classifier[-1].in_features #Stores the input parameters that are comong from the last second layer(ie. in this case they are 4096)

    del pre_model.model.classifire[-1] #Deletes the last layer

    shared_model_path= os.path.join(os.getcwd(),"models","shared_model.pth")
    path_to_reg = os.path.join(os.getcwd(),"models","reg_params.pickle")
    if(os.path.exists(shared_model_path)):
        model.load_state_dict(torch.load(shared_model_path))

    ## Adding the new classification head to the shared model
    pre_model.model.classifier.add_module('6', nn.Linear(in_features,no_of_classes))

    ## Loading the reg_params stored
    if os.path.isfile(path_to_reg):
        with open(path_to_reg,'rb') as handle:
            reg_params = pickle.load(handle)

        premodel.params = reg_params

    device = torch.device("cuda:0" if use_gpu else "cpu")

    pre_model.train(True)
    pre_model.to(device)

    return pre_model

def forgetting(task,dataloader,size,use_gpu)
    """
    Funtion to calculate the forgetting on previous tasks whuch have already been learnt
    """