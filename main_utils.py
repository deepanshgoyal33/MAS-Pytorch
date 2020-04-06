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
from optimizers import *

def model_initialiser(no_of_classes,use_gpu):

    """
    We have to delete the classifiaction head of the model coming from the previous task so we take the model   
    and detach its classification head and put a new one and initialise another model with new classification head
    but the features are shared.
    Inputs-

    Output-

    """
    init_model = models.alexnet(pretrained = True)
    model= SharedModel(init_model)

    ## New classifiaction head
    in_features = model.xmodel.classifier[-1].in_features #Stores the input parameters that are comong from the last second layer(ie. in this case they are 4096)

    del model.xmodel.classifire[-1] #Deletes the last layer

    shared_model_path= os.path.join(os.getcwd(),"models","shared_model.pth")
    path_to_reg = os.path.join(os.getcwd(),"models","reg_params.pickle")
    if(os.path.exists(shared_model_path)):
        model.load_state_dict(torch.load(shared_model_path))

    ## Adding the new classification head to the shared model
    model.xmodel.classifier.add_module('6', nn.Linear(in_features,no_of_classes))

    ## Loading the reg_params stored
    if os.path.isfile(path_to_reg):
        with open(path_to_reg,'rb') as handle:
            reg_params = pickle.load(handle)

        model.params = reg_params

    device = torch.device("cuda:0" if use_gpu else "cpu")

    model.train(True)
    model.to(device)

    return model

def MAS(model,task,epochs,no_of_classes,lr=.001,schduler_lambda=.01,num_frozen,use_gpu=False,trdataload,tedataload,train_size,test_size):
    """
    Training Loop for the MAS
    Inputs:
        model
        task
        epochs
        no_of_classes
        lr
        scheduler_lambda
        use_gpu
        trdataload
        tedataload
        train_size
        test_size

    Outputs:
        model : Trained model
    """
    ## For task no. 1
    if (task ==1):
        model,freezed_layers = create_freeze_layers(model,num_frozen)
        
        model = initialsing_regulariser(model, use_gpu, freezed_layers,task)

    else:
        device = torch.device("cuda:0" if use_gpu else "cpu")
        ## Now our model would have trained for task 1 by now we have to get the params learnt from previous task and for 
        ## for the num of layers that are frezon we have to reinitialise the omega prameters
        reg_params = model.params

        for name, param in model.xmodel.named_parameters():

            if not name in freezed_layers:

                if param in reg_params:

                    param_dict = reg_params[param]

                    print("Initialising omega values for {} layer in {} task".format(name,task))
                    ## previous values of omega
                    prev_omega = parma_dict['omega'] 
                    new_omega = torch.zeros(param.size())
                    new_omega = omega.to(device)
                    init_val = prama.data.clone()
                    init_val = init_val.to(device)
                    param_dict["prev_omega"]= prev_omega
                    parma_dict['omega'] = new_omega
                    #storing the initial values of the parameters
                    param_dict['init_val']= init_val
                    reg



def compute_forgetting(task,dataloader,size,use_gpu)
    """
    Funtion to calculate the forgetting on previous tasks whuch have already been learnt
    """
    