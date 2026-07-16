import torch
import torch.nn as nn
from config import *

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNEL,CONV1_OUT,kernel_size=KERNEL_SIZE,padding = 1)
        self.conv2 = nn.Conv2d(CONV1_OUT,CONV2_OUT,kernel_size=KERNEL_SIZE,padding = 1)
        self.pool  = nn.MaxPool2d(POOL_KERNEL,POOL_KERNEL)
        self.fc1   = nn.Linear(CONV2_OUT*7*7,FC1_UNITS)
        self.drop  = nn.Dropout(DROPOUT)
        self.fc2   = nn.Linear(FC1_UNITS,NUM_CLASS) 
        self.relu  = nn.ReLU()
    
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)

torch.manual_seed(SEED)
net = CNN()
