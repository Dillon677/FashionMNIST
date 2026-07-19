import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from model import CNN
from dataloader import train_loader, valid_loader

device = DEVICE
net = CNN().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

best_val_loss = float('inf')
patience_counter = 0
patience = 8

for epoch in range(1, EPOCHES + 1):
    net.train()
    train_loss, train_correct, train_seen = 0, 0, 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        out = net(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * xb.size(0)
        train_correct += (out.argmax(1) == yb).sum().item()
        train_seen += xb.size(0)
    net.eval()
    valid_loss, valid_correct, valid_seen = 0, 0, 0
    
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = net(xb)
            loss = criterion(out, yb)
            
            valid_loss += loss.item() * xb.size(0)
            valid_correct += (out.argmax(1) == yb).sum().item()
            valid_seen += xb.size(0)
    
    train_acc = train_correct / train_seen
    valid_acc = valid_correct / valid_seen
    valid_loss_avg = valid_loss / valid_seen
    
    print(f"Epoch {epoch:3d} | train_loss {train_loss/train_seen:.4f} "
          f"train_acc {train_acc*100:6.2f}% | valid_loss {valid_loss_avg:.4f} "
          f"valid_acc {valid_acc*100:6.2f}%")