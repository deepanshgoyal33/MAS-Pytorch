import torch 
import torch.nn as nn
import os
import shutil
import torchvision
from torchvision import datasets, models, transforms

"The classification Head remains differest which is task based"

"class specific features are limited to last layer"
def ClassHead(nn.Module):
    """
    Only the last layer changes which is task specific 
    """
    def __init__(self,in_features,output_features):
        super(ClassHead,self).__init__()
        self.classhead = nn.Linear(in_features, output_features)

    def forward(self, x):
        return x



def SharedModel(nn.Module):
    """
    As Shared model is same all across, I am taking alexnet for time being as the baseline
    """
    def __init__(self):
        super(SharedModel, self).__init__()
        self.model = models.alexnet(pretrained=True)
        self.params = {}

    def forward(self,x):
        return self.model(x)
