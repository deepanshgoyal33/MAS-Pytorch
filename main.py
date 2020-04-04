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

def train():





def test():


if __name__="__main__":

    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("--batch_size", default= 16, type= int, help="The batch size you want to use")
    parser.add_argument("--epochs", default = 50, type = int, help="no of epochs you want to run for")
    parser.add_argument("--num_freezed",default = 2, type= int, help="No of Layers you want to freeze while extracting features")
    parser.add_argument("--lr", default=.001, type= float, help = 'The initial Learning Rate you want to set')
    parser.add_argument("--lamda", default =.01, type = float, help= "Regularisation Parameter")
    args= parser.parse_args()


