import torch
from torch import nn, optim
from torch.utils import data
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import os
import sys
import cv2
import seaborn as sns

def test_basic():
    t1 = torch.tensor([[1, 2], [4, 7]]).to(dtype=torch.float16)
    t2 = torch.randn(size=(1, 2, 4))
    print(t1[0, :])
    print(t2.max().item())
    print((t1 - t1.mean())[0].std().item(), (t1 - t1.mean())[1].std().item())
    hwc_tensor = torch.rand((640, 480, 3))
    chw_tensor = hwc_tensor.permute(2, 0, 1)
    #chw_tensor.clamp(100, 1200)
    print(chw_tensor.shape)

def torch_dataloader():
    imagenet = datasets.Ca
    train_data_path = "/train"
    train_data = torchvision.datasets.ImageFolder(
        root = train_data_path, transform=transforms
    )
    val_data_path = "/val"
    val_data = torchvision.datasets.ImageFolder(
        root = val_data_path, transform=transforms
    )
    test_data_path = "/test"
    test_data = torchvision.datasets.ImageFolder(
        root = test_data_path, transform=transforms
    )
    batch_size = 64
    train_date_loader = data.DataLoader(train_data, batch_size)
    val_date_loader = data.DataLoader(val_data, batch_size)
    test_date_loader = data.DataLoader(test_data, batch_size)

class SimpleNet(nn.Module):
    "a simple first neural net"
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.l1 = nn.Linear(10000, 100)
        self.l2 = nn.Linear(100, 20)
        self.l3 = nn.Linear(20, 2)

    def forward(self):
        x = x.view(-1, 10000)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.softmax(self.l3(x))
        return x

if __name__=="__main__":
    transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[1.0], std=[0.2])
    ])
    test_basic()
    mynet = SimpleNet()
    print(mynet)