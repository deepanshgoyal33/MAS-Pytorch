import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import argparse 
import numpy as np
from random import shuffle
import copy
import sys 

from utils import *
from model_utils import *


def train():





def test():


if __name__="__main__":

    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("--batch_size", default= 16, type= int, help="The batch size you want to use")
    parser.add_argument("--epochs", default = 50, type = int, help="no of epochs you want to run for")
    parser.add_argument("--num_freezed",default = 2, type= int, help="No of Layers you want to freeze while extracting features")
    parser.add_argument("--lr", default=.001, type= float, help = 'The initial Learning Rate you want to set')
    parser.add_argument("--lamda", default =.01, type = float, help= "Regularisation Parameter")
    parser.add_argument("--use_gpu",default= False,type= bool, help="if to use gpu while training or not")
    args= parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    num_frozen = args.num_freezed
    lr = args.lr
    scheduler_lambda = args.lamda

    path= os.getcwd()

    train_dataloaders = []
    test_dataloaders = []

    train_datasets =[]
    test_datasets=[]

    num_classes=[]

    path= os.getcwd()
    datapath= os.path.join(path,"data")

    transforms={
        'train_transforms': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ]),
        'test_transforms': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
    }
    # split the data into the number  tasksyou want to before this step
    # for our case these are 4
    # 4 folders in Data folder with no as their task number 
    for ndir in sorted(os.listdir(data_dir)):
        
        ##objects for the train and test image folder
        train_image_folder = datasets.ImageFolder(os.path.join(datapath,ndir,"train"),transform = transforms['train_transforms'])
        test_image_folder = datsets.ImageFolder(os.path.join(dataset,dir,'test'),transform = transforms['test_tranforms'])

        ## Loading the data into batches
        train_dataloader= torch.utils.data.Dataloader(train_image_folder, batch_size= batch_size, shuffle= True,num_workers = 3)
        test_dataloader = torch.utils.data.Dataloader(test_image_folder, batch_size= batch_size,shuffle= True,num_workers = 1)

        train_size = len(train_image_folder)
        test_size = len(test_image_folder)

        ##appending the datloader that we have created just now for future use
        train_dataloaders.append(train_dataloader)
        test_dataloaders.append(test_dataloader)

        