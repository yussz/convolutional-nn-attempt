#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import sys

w,h = 28,28
n_classes=3
device = "cuda"
learning_rate = 1e-2
n_labels =3
epochs =10

transform =  transforms.Compose([
    transforms.Resize((w,h)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(root="./afhq/train",transform=transform)
train_dataloader = DataLoader(train_data,batch_size=64, shuffle=True,num_workers=4)

test_data = datasets.ImageFolder(root="./afhq/val",transform=transform)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True,num_workers=4)

# iter = iter(test_dataloader)
# images, labels = next(iter)
# print(labels)
# sys.exit()

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,1,1) 
        self.pool1 = nn.MaxPool2d(1,1) 

        self.conv2 = nn.Conv2d(16,32,3,1,1)
        self.pool2 = nn.MaxPool2d(1,1)

        self.fc1 = nn.Linear(28*28*32,256)
        self.fc2 = nn.Linear(256,n_labels)

    def forward(self,x,targets=None):
        #64,3,28,28
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        B,C,H,W = x.shape
        x = x.view(B,C*H*W)
        x = self.fc1(x)
        x = self.fc2(x)
        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(x,targets)
        return x, loss
        


torch.manual_seed(19)
model =CNN()
model = model.to(device)
print(sum(p.numel() for p in model.parameters()), "params")
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)


def train(dataloader):
    model.train()
    num_batches = len(dataloader)
    for batch,(x,y) in enumerate(dataloader):
        x,y = x.to(device),y.to(device)
        pred,loss = model(x,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"batch [{batch}:{num_batches}], loss: {loss.item()}") if batch % 8 == 0  else None

@torch.no_grad()
def test(dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss,total_correct = 0, 0
    for x,y in dataloader:
        x, y = x.to(device), y.to(device)
        pred,loss = model(x,y)
        avg_loss += loss.item()
        total_correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
    avg_loss /= num_batches
    total_correct /=size
    print(f"Test Error: \n Accuracy: {(100*total_correct):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

def main():
    for epoch in range(epochs+1):
        print(f"epoch: {epoch} {'--' *20}")
        train(train_dataloader)
        test(test_dataloader)

main()

   
