import torch
from torchvision import datasets,transforms
from torch.utils.data import random_split,DataLoader
from config import *

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_full = datasets.FashionMNIST(root="./data",train=True,download=True,transform=tf)
test_set   = datasets.FashionMNIST(root="./data",train=False,download=True,transform=tf)

g = torch.Generator().manual_seed(SEED)
train_set,valid_set = random_split(train_full,[50000,10000],generator=g)

train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
valid_loader = DataLoader(valid_set,batch_size=BATCH_SIZE,shuffle=False)
test_loader  = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False)
