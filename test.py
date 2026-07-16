import torch
import torch.nn as nn
from config import *
from model import net
from dataloader import test_loader

net.load_state_dict(torch.load('best_model.pth'))
net.eval()

criterion = nn.CrossEntropyLoss()

test_loss,test_acc,seen = 0,0,0
with torch.no_grad():
    for xb,yb in test_loader:
        out = net(xb)
        test_loss += criterion(out,yb).item() * xb.size(0)
        test_acc  += (out.argmax(1) == yb).sum().item()
        seen      += xb.size(0)
    
avg_loss = test_loss/seen
accuracy = test_acc/seen

print(f"测试集损失: {avg_loss:.4f}")
print(f"测试集准确率: {accuracy*100:.2f}%")
print(f"共 {seen} 张测试图片，其中错了 {seen - test_acc} 张")