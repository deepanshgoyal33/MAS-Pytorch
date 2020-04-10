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
import time


from optimizers import *
from main_utils import *
from MAS_utils import *

def mas_train(model,optimizer, model_criterion,task,epochs,no_of_classes,lr,scheduler_lambda,num_frozen,use_gpu,trdataload,tedataload,train_size,test_size):
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
        if(checkpoint_file==""):
            start_epoch = 0
        else:
            print("Loading checkpoint '{}' ".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            start_epoch = checkpoint['epoch']
            print("Loading the model")
            ##Initialises the model with last classifier layers chaged as paer our needs and weights of the shared model put inside the model
            model = model_initialiser(no_of_classes,use_gpu)
            model = model.load_state_dict(checkpoint['state_dict'])

            print('Loading the optimizer')
            optimizer = local_sgd(model.params, scheduler_lambda)
            optimizer = optimizer.load_state_dict(checkpoint['optimizer'])

            print('Done')

    model.xmodel.train(True)
    model.xmodel.to(device)

    #training Loop starts
    for epoch in range(start_epoch,epochs+1):
        ## Omega accumulation is done at the convergence of the loss function
        if(epoch == epochs):
            ## Notice the fact that no training happens during this 
            optimizer_ft = omega_update(model.params)
            print("Updating the omega values for this task")
            ## takes the input images calculate gradient and upadte the params
            model = compute_omega_grads_norm(model,trdataload,optimizer_ft,use_gpu)

            running_loss = 0
            running_corrects=0
            model.tmodel.eval()
            for data in dataloader_test:
                input_data , labels = data
                del data
                if use_gpu:
                    input_data = input_data.to(device)
                    labels = labels.to(device)
                else:
                    input_data  =  input_data
                    labels = Variable(labels)
				#optimizer.zero_grad()
                output = model.tmodel(input_data)
                del input_data
                _, preds = torch.max(output, 1)
                del output
                
                running_corrects += torch.sum(preds == labels.data)
                del preds
                del labels
                epoch_accuracy = running_corrects.double()/dset_size_test

        else:

            since = time.time()
            best_perform = 10e6

            print("Training on epoch no {} of {}".format(epoch+1,num_epoch))
            print("-"*20)
            running_loss = 0
            running_corrects = 0

            ## returning the optimizer and making it smaller every 20 rounds
            optimizer = scheduler(optimizer, epoch,lr)

            model.xmodel.train(True)

            for input_data,labels in dataloader_train:
                if use_gpu:
                    input_data = input_data.to(device)
                    labels = labels.to(device)

                else:
                    ## variable is just a wrapper around the tensors
                    input_data = Variable(input_data)
                    labels = Variable(labels)


                model.xmodel.to(device)
                ## resets the gradients
                optimizer.zero_grad()

                output = model.xmodel(input_data)
                del input_data

                not_req, predictions = torch.max(output,1)
                loss = model_criterion(output,labels)
                
                del output
                ##automatically computes the gradients and changes the parameters jfor which requires_grad is True

                loss.backward()
                optimizer.step(model.params)

                running_loss += loss.item()
                del loss
                running_corrects += torch.sum(preds = labels.data)
                del preds 
                del labels

            epoch_loss = running_loss/train_size
            ## In order to get the accuracy in double we need to have atleast 1 variable of type double
            epoch_accuracy =  running_corrects.double()/train_size

            print("Loss: {} Accuracy:{} ".format(epoch_loss,epoch_accuracy))
            
            # avoiding the filw to be written twice
            if(epoch!=0 and epoch != epochs-1 and (epoch+1)%10 ==0):
                epoch_file = os.path.join(store_path, str(epoch+1),".pth.tar")
                torch.save({
                    'epoch': epoch,
                    'epoch_loss': epoch_loss,
                    'epoch_accuracy':epoch_accuracy,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, epoch_file_name)
    save_model(model, task, epoch_accuracy)



