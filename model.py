import torch
import torch.nn as nn
from config import *

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNEL,CONV1_OUT,kernel_size=KERNEL_SIZE,padding = 1)
        self.bn1   = nn.BatchNorm2d(CONV1_OUT)
        self.conv2 = nn.Conv2d(CONV1_OUT,CONV2_OUT,kernel_size=KERNEL_SIZE,padding = 1)
        self.bn2   = nn.BatchNorm2d(CONV2_OUT)
        self.conv3 = nn.Conv2d(CONV2_OUT,CONV3_OUT,kernel_size=KERNEL_SIZE,padding = 1)
        self.bn3   = nn.BatchNorm2d(CONV3_OUT)
        self.pool  = nn.MaxPool2d(POOL_KERNEL,POOL_KERNEL)

        self.fc1   = nn.Linear(CONV3_OUT*3*3,FC1_UNITS)
        self.drop1 = nn.Dropout(DROPOUT)
        self.fc2   = nn.Linear(FC1_UNITS,FC2_UNITS) 
        self.drop2 = nn.Dropout(DROPOUT)
        self.fc3   = nn.Linear(FC2_UNITS,NUM_CLASS)
        self.relu  = nn.ReLU()
    
    def forward(self,x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        return self.fc3(x)

torch.manual_seed(SEED)
net = CNN()
