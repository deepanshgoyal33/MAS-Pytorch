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


from optimizers import *
from main_utils import *
from MAS_utils import *

def mas_train(model,optimizer, model_criterion,task,epochs,no_of_classes,lr=.001,scheduler_lambda=.01,num_frozen,use_gpu=False,trdataload,tedataload,train_size,test_size):
    """
    Training Loop of the model

    """

    omega_epochs = epochs+1
    store_path = os.path.join(os.getcwd(),"models","Task_"+str(task))
    model_path = os.path.join(os.getcwd(),"models")
    device = torch.device("cuda:0" if use_gpu else "cpu")

    ## Creating a directory if there is no dircetory
    if(task== 1 and not os.path.isdir(model_path)):
        os.mkdir(model_path)

    checkpoint_file, flag = check_checkpoints(store_path)
    
    if(flag == False):
        create_task_dir( no_of_classes, store_path)
        start_epoch =0

    else:
        print("Loading checkpoint '{}' ".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        print("Loading the model")
        ##Initialises the model with last classifier layers chaged as paer our needs and weights of the shared model put inside the model
        model = model_initialiser(no_of_classes,use_gpu)
        model = model.load_state_dict(checkpoint['state_dict'])

        print('Loading the optimizer')
        optimizer = local_sgd(model.reg_params, reg_lambda)
        optimizer = optimizer.load_state_dict(checkpoint['optimizer'])

        print('Done')

    model.xmodel.train(True)
    model.xmodel.to(device)

    #training Loop starts
    for epoch in range(start_epoch,epochs+1):

        ## Omega accumulation is done at the convergence of the loss function
        if(epoch == epochs-1):
            ## Notice the fact that no training happens during this 
            optimizer_
