import sys
import argparse
from MAS_utils import *
from main_utils import *
import copy
import torch.utils.data
from random import shuffle
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch
torch.backends.cudnn.benchmark = True



parser = argparse.ArgumentParser(description="Testing")
parser.add_argument("--batch_size", default=16, type=int,
                    help="The batch size you want to use")
parser.add_argument("--epochs", default=50, type=int,
                    help="no of epochs you want to run for")
parser.add_argument("--num_freezed", default=2, type=int,
                    help="No of Layers you want to freeze while extracting features")
parser.add_argument("--lr", default=.008, type=float,
                    help='The initial Learning Rate you want to set')
parser.add_argument("--lamda", default=.01, type=float,
                    help="Regularisation Parameter")
parser.add_argument("--use_gpu", default=torch.cuda.is_available(),
                    type=bool, help="if to use gpu while training or not")
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
num_frozen = args.num_freezed
lr = args.lr
scheduler_lambda = args.lamda
use_gpu = args.use_gpu

path = os.getcwd()

train_dataloaders = []
test_dataloaders = []

train_datasets = []
test_datasets = []

num_classes = []

path = os.getcwd()
datapath = os.path.join(path, "Data")

transforms = {
    'train_transforms': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test_transforms': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
# split the data into the number  tasksyou want to before this step
# for our case these are 4
# 4 folders in Data folder with no as their task number
for ndir in sorted(os.listdir(datapath)):

    # objects for the train and test image folder
    train_image_folder = datasets.ImageFolder(os.path.join(
        datapath, ndir, "train"), transform=transforms['train_transforms'])
    test_image_folder = datasets.ImageFolder(os.path.join(
        datapath, ndir, "test"), transform=transforms['test_transforms'])

    # Loading the data into batches
    train_dataloader = torch.utils.data.DataLoader(
        train_image_folder, batch_size=batch_size, shuffle=True, num_workers=3)
    test_dataloader = torch.utils.data.DataLoader(
        test_image_folder, batch_size=batch_size, shuffle=True, num_workers=1)

    train_size = len(train_image_folder)
    test_size = len(test_image_folder)

    num_classes.append(len(train_image_folder.classes))

    # appending the datloader that we have created just now for future use
    train_dataloaders.append(train_dataloader)
    test_dataloaders.append(test_dataloader)

    train_datasets.append(train_size)
    test_datasets.append(test_size)


no_tasks = len(train_dataloaders)
print(no_tasks)

# initialising the model
model = SharedModel(models.alexnet(pretrained=True))

# model training on the given no of tasks
for task in range(1, no_tasks+1):
    print("Training starting on the task #{}".format(task))
    trdataloader = train_dataloaders[task-1]
    tedataloader = test_dataloaders[task-1]
    train_size = train_datasets[task-1]
    test_size = test_datasets[task-1]

    no_of_classes = num_classes[task-1]

    # refer to the model utils
    model = model_initialiser(no_of_classes, use_gpu)

    # We have the initialised model and reg params inside the model

    MAS(model, task, epochs, no_of_classes, lr, scheduler_lambda,
        num_frozen, use_gpu, trdataloader, tedataloader, train_size, test_size)

print("The training process on all the tasks are completed")

print("Testing Phase")

# test the sets on the test sets of the tasks

for task in range(1, no_tasks + 1):
    print("Testiong the model on task{}".format(task))
    dataloader = test_dataloaders[task-1]
    dset_size = train_datasets[task-1]
    no_of_classes = num_classes[task-1]
    forgetting = compute_forgetting(task, dataloader, dset_size, use_gpu)
    print("The forgetting undergone on task {} is {}".format(task, forgetting))
