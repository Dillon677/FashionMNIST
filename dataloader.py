import torch
from torchvision import datasets,transforms
from torch.utils.data import random_split,DataLoader
from config import *

train_tf = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(AUG_ROTATION_DEGREES),       
    transforms.RandomAffine(0, translate=(AUG_TRANSLATE_RATIO, AUG_TRANSLATE_RATIO)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
    transforms.RandomErasing(p=0.2),      
])

eval_tf= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_full = datasets.FashionMNIST(root=DATA_PATH, train=True,  download=True, transform=None)
test_set   = datasets.FashionMNIST(root=DATA_PATH, train=False, download=True, transform=eval_tf)

g = torch.Generator().manual_seed(SEED)
train_set,valid_set = random_split(train_full,[50000,10000],generator=g)

train_set.dataset.transform = train_tf
valid_set.dataset.transform = eval_tf

train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
valid_loader = DataLoader(valid_set,batch_size=BATCH_SIZE,shuffle=False)
test_loader  = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False)
