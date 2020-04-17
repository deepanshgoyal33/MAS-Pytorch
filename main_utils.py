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

from MAS_model import *
from optimizers import *
from trainloop import *


def model_initialiser(no_of_classes, use_gpu):
    """
    We have to delete the classifiaction head of the model coming from the previous task so we take the model   
    and detach its classification head and put a new one and initialise another model with new classification head
    but the features are shared.
    Inputs-

    Output-

    """
    init_model = models.alexnet(pretrained=True)
    model = SharedModel(init_model)

    # New classifiaction head
    # Stores the input parameters that are comong from the last second layer(ie. in this case they are 4096)
    in_features = model.xmodel.classifier[-1].in_features

    del model.xmodel.classifier[-1]  # Deletes the last layer

    shared_model_path = os.path.join(os.getcwd(), "models", "shared_model.pth")
    path_to_reg = os.path.join(os.getcwd(), "models", "reg_params.pickle")
    if(os.path.exists(shared_model_path)):
        model.load_state_dict(torch.load(shared_model_path))

    # Adding the new classification head to the shared model
    model.xmodel.classifier.add_module(
        '6', nn.Linear(in_features, no_of_classes))

    # Loading the reg_params stored
    if os.path.isfile(path_to_reg):
        with open(path_to_reg, 'rb') as handle:
            reg_params = pickle.load(handle)

        model.params = reg_params

    device = torch.device("cuda:0" if use_gpu else "cpu")

    model.train(True)
    model.to(device)

    return model


def MAS(model, task, epochs, no_of_classes, lr, scheduler_lambda, num_frozen, use_gpu, trdataload, tedataload, train_size, test_size):
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
    # For task no. 1
    if (task == 1):
        model, freezed_layers = create_freeze_layers(model, num_frozen)

        model = initialsing_omega(model, use_gpu, task, freezed_layers)

    else:
        device = torch.device("cuda:0" if use_gpu else "cpu")
        # Now our model would have trained for task 1 by now we have to get the params learnt from previous task and for
        # for the num of layers that are frezon we have to reinitialise the omega prameters
        reg_params = model.params

        for name, param in model.xmodel.named_parameters():

            if not name in freezed_layers:

                if param in reg_params:

                    param_dict = reg_params[param]

                    print(
                        "Initialising omega values for {} layer in {} task".format(name, task))
                    # previous values of omega
                    prev_omega = parma_dict['omega']
                    new_omega = torch.zeros(param.size())
                    new_omega = omega.to(device)
                    init_val = prama.data.clone()
                    init_val = init_val.to(device)
                    param_dict["prev_omega"] = prev_omega
                    parma_dict['omega'] = new_omega
                    # storing the initial values of the parameters
                    param_dict['init_val'] = init_val
                    reg_params[param] = param_dict

        model.reg_params = reg_params

    # model and omega values created
    # optimizers
    model_criterion = nn.CrossEntropyLoss()
    optimizer = local_sgd(model.xmodel.parameters(), scheduler_lambda, lr)
    mas_train(model, optimizer, model_criterion, task, epochs, no_of_classes, lr,
              scheduler_lambda, num_frozen, use_gpu, trdataload, tedataload, train_size, test_size)


def compute_forgetting(task, dataloader, size, use_gpu):
    """
    Funtion to calculate the forgetting on previous tasks whuch have already been learnt
    """
    # Getting the trained files
    store_path = os.path.join(os.getcwd(), "models", "Task_"+str(task))
    model_path = os.path.join(os.getcwd(), "models")
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # get the old performance
    file_object = open(os.path.join(store_path, "performance.txt"), 'r')
    old_performance = file_object.read()
    file_object.close()

    model = model_inferece(task_no, use_gpu=False)
    model.to(device)
    running_corrects = 0
    for data in dataloader:
        input_data, labels = data
        del data
        if (use_gpu):
            input_data, labels = input_data.to(device), labels.to(device)

        else:
            input_data = Variable(input_data)
            labels = Variable(labels)

        output = model.xmodel(input_data)
        del input_data

        _, preds = torch.max(output, 1)

        running_corrects += torch.sum(preds == labels.data)
        del preds
        del labels

    epoch_accuracy = running_corrects.double()/dset_size
    old_performance = float(old_performance)
    forgetting = epoch_accuracy.item() - old_performance
    return forgetting
