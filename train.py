import torch
import torch.nn as nn
import torch.optim as optim
import time
from config import *
from model import net
from dataloader import train_loader, valid_loader

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

best_val_acc = 0.0

for epoch in range(1, EPOCHES + 1):
    net.train()
    t0 = time.time()
    running_loss, correct, seen = 0, 0, 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = net(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        seen += xb.size(0)

    train_loss, train_acc = running_loss / seen, correct / seen

    net.eval()
    vloss, vcorrect, vseen = 0, 0, 0
    with torch.no_grad():
        for xb, yb in valid_loader:
            out = net(xb)
            vloss += criterion(out, yb).item() * xb.size(0)
            vcorrect += (out.argmax(1) == yb).sum().item()
            vseen += xb.size(0)

    valid_loss, valid_acc = vloss / vseen, vcorrect / vseen

    print(f"Epoch {epoch}/{EPOCHES} | "
          f"train_loss {train_loss:.4f} train_acc {train_acc*100:.2f}%  | "
          f"valid_loss {valid_loss:.4f} valid_acc {valid_acc*100:.2f}%  | "
          f"{time.time()-t0:.1f}s")

    # 保存验证集上表现最好的模型
    if valid_acc > best_val_acc:
        best_val_acc = valid_acc
        torch.save(net.state_dict(), 'best_model.pth')
        print(f"  → 保存最佳模型 (valid_acc={valid_acc*100:.2f}%)")

print(f"\n训练完成,最佳验证准确率: {best_val_acc*100:.2f}%")