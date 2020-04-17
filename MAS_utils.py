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
import pickle
from torch.utils.data import DataLoader

from optimizers import *
from MAS_model import *

def create_freeze_layers(model, num_layers=2):
    """
    Returning the model with frozen layers for which require gradient argument to be false
    Inputs:
        model= 
        num_layers = No. of frozen layers
    Output:
        model, frozen layers
    """
    # We want features no to be changed only the classifier learning according to tasks
    for param in model.xmodel.classifier.parameters():
        param.require_grad = True

    for param in model.xmodel.features.parameters():
        param.require_grad = False

    # returning a n empty list of the num o layeers is zero
    if(num_layers == 0):
        return []

    temp_list = []
    frozen_layers_data = []
    # If we want 2 layers frozen then we have to make the requires_grad False for the num_layers number of conv2d in the feature module of the alexnet
    # and we can do that by making all other's requires_grad True
    for key in model.xmodel.features._modules:
        if(type(model.xmodel.features._modules[key]) == torch.nn.modules.conv.Conv2d):
            temp_list.append(key)

    no_non_frozen_layers = len(temp_list)-num_layers

    for no in range(0, no_non_frozen_layers):
        temp_key = temp_list[no]
        for param in model.xmodel.features[int(temp_key)].parameters():
            param.requires_grad = True

        name1 = "features." + temp_key + ".weight"
        name2 = "features." + temp_key + ".bias"

        # this list will contatin the name of the layers for which requires_grad is made true
        frozen_layers_data.append(name1)
        frozen_layers_data.append(name2)

    return model, frozen_layers_data


def initialsing_omega(model, use_gpu,task,frozen_layers=[]):
    """
    Function:

        Inputs:

        Outputs:

    """
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # A dictionary to store the initialised omega values with the key and the values being the weights of the layer and importance value
    reg_params = {}

    # Iterating over the the model in which name will give the name of the layer and param is giving the weights stored inside the correseponding layer
    # For the frozen layers the omega will be calculated and reset every time new task is being trained
    for name, param in model.xmodel.named_parameters():
        if not name in frozen_layers:

            print("Initialsing omega values for layer", name)
            omega = torch.zeros(param.size())
            omega = omega.to(device)

            init_val = param.data.clone()
            param_dict = {}

            # init_val is just a variable which stores the weights of the corresponding layer
            # for first task, omega is initialised to zero
            param_dict["omega"] = omega
            param_dict["init_val"] = init_val  # storing the initial value

            reg_params[param] = param_dict

    model.params = reg_params

    return model


def check_checkpoints(storepath):
    if not os.path.exists(storepath):
        checkpoint_file = ""
        flag = False

    # directory exists but there is no checkpoint file
        onlyfiles = [f for f in os.listdir(
            storepath) if os.path.isfile(os.path.join(store_path, f))]
        max_train = -1
        flag = False

        # Check the latest epoch file that was created
        for file in onlyfiles:
            if(file.endswith('pth.tr')):
                flag = True
                test_epoch = file[0]
                if(test_epoch > max_train):
                    max_epoch = test_epoch
                    checkpoint_file = file

        # no checkpoint exists in the directory so return an empty string
        if (flag == False):
            checkpoint_file = ""

    return checkpoint_file, flag


def create_task_dir(num_classes, store_path):
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    file_path = os.path.join(store_path, "classes.txt")
    with open(file_path, 'w') as file1:
        input = str(num_classes)
        file1.write(input)
        file1.close()

    return


def scheduler(optimizer, epoch, lr=.0008):
    """
    Function: This function will decay the learning rate after every 20 epochs
    Inputs:
        optimizer: Localsgd in our case

    """
    weight_decay_epoch = 20
    lr = lr * (.1 ** (epoch // weight_decay_epoch))
    print("lr is "+str(lr))

    if (epoch % weight_decay_epoch == 0):
        print("Lr is set to {}".format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def save_model(model, task, accuracy):
    """
    Function to save the model
    """
    path_to_model = os.path.join(os.getcwd(), "models")
    path_to_head = os.path.join(path_to_model, "Task_"+ str(task))

    in_features = model.xmodel.classifier[-1].in_features
    out_features = model.xmodel.classifier[-1].out_features

    ref = ClassHead(in_features, out_features)
    ref.classhead.weight.data = model.xmodel.classifier[-1].weight.data
    ref.classhead.bias.data = model.xmodel.classifier[-1].bias.data

    # params a;ready saved in the params object inside the model class
    reg_params = model.params

    f = open(os.path.join(os.getcwd(), "models", "reg_params.pickle"), 'wb')
    pickle.dump(reg_params, f)
    f.close()

    del model.xmodel.classifier[-1]

    torch.save(model.state_dict(), os.path.join(
        os.getcwd(), "models", "shared_model.pth"))
    torch.save(ref.state_dict(), os.path.join(
        os.getcwd(), "models", "Task_"+str(task), "head.pth"))

    with open(os.path.join(path_to_head, "performance.txt"), 'w') as inputfile:
        input = str(accuracy.item())
        inputfile.write(input)
        inputfile.close()

    del ref


def compute_omega_grads_norm(model, dataloader, optimizer, use_gpu):
    """

    """
    # Setting the model on evaluation mode
    model.xmodel.eval()

    index = 0
    for inputs, labels in dataloader:

        if(use_gpu):
            device = torch.device('cuda' if use_gpu else "cpu")
            inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model.xmodel(inputs)
        del inputs

        l2_norm = torch.norm(outputs, 2, dim=1)
        del outputs
        squared_l2_norm = l2_norm**2
        del l2_norm

        sum_norm = torch.sum(squared_l2_norm)
        del squared_l2_norm

        sum_norm.backward()

        optimizer.step(model.params, index, labels.size(0), use_gpu)
        del labels
        index = index+1

    return model


def compute_omega_grads_vector(model, dataloader, optimizer, use_gpu):
    """
    Inputs:
    1) model: A reference to the model for which omega is to be calculated
    2) dataloader: A dataloader to feed the data to the model
    3) optimizer: An instance of the "omega_update" class
    4) use_gpu: Flag is set to True if the model is to be trained on the GPU

    Outputs:
    1) model: An updated reference to the model is returned

    Function: This function backpropagates across the dimensions of the  function (neural network's) 
    outputs. In addition to this, the function also accumulates the values of omega across the items 
    of a task. Refer to section 4.1 of the paper for more details regarding this idea

    """

    # Alexnet object
    model.tmodel.train(False)
    model.tmodel.eval(True)

    index = 0

    for dataloader in dset_loaders:
        for data in dataloader:

                # get the inputs and labels
            inputs, labels = data

            if(use_gpu):
                device = torch.device("cuda:0")
                inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # get the function outputs
            outputs = model.tmodel(inputs)

            for unit_no in range(0, outputs.size(1)):
                ith_node = outputs[:, unit_no]
                targets = torch.sum(ith_node)

                # final node in the layer
                if(node_no == outputs.size(1)-1):
                    targets.backward()
                else:
                    # This retains the computational graph for further computations
                    targets.backward(retain_graph=True)

                optimizer.step(model.reg_params, False,
                               index, labels.size(0), use_gpu)

                # necessary to compute the correct gradients for each batch of data
                optimizer.zero_grad()

            optimizer.step(model.reg_params, True, index,
                           labels.size(0), use_gpu)
            index = index + 1

    return model


# sanity check for the model to check if the omega values are getting updated
def sanity_model(model):

    for name, param in model.tmodel.named_parameters():

        print(name)

        if param in model.reg_params:
            param_dict = model.reg_params[param]
            omega = param_dict['omega']

            print("Max omega is", omega.max())
            print("Min omega is", omega.min())
            print("Mean value of omega is", omega.min())


def consolidate_reg_params(model, use_gpu):
    """
    Input:
    1) model: A reference to the model that is being trained
    2) use_gpu: Set the flag to True if you wish to train the model on a GPU

    Output:
    1) reg_params: A dictionary containing importance weights (omega), init_val (keep a reference 
    to the initial values of the parameters) for all trainable parameters


    Function: This function updates the value (adds the value) of omega across the tasks that the model is 
    exposed to

    """
    # Get the reg_params for the model
    reg_params = model.params

    for name, param in model.xmodel.named_parameters():
        if param in reg_params:
            param_dict = reg_params[param]
            print("Consolidating the omega values for layer", name)

            # Store the previous values of omega
            prev_omega = param_dict['prev_omega']
            new_omega = param_dict['omega']

            new_omega = torch.add(prev_omega, new_omega)
            del param_dict['prev_omega']

            param_dict['omega'] = new_omega

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

    model.reg_params = reg_params

    return mode


