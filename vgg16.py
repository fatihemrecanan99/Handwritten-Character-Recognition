import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
# kfold
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

import torch.optim as optim
import pandas as pd
import numpy as np

import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


# vgg16
class VGG16(nn.Module):
    def _init_(self, num=26):
        super(VGG16, self)._init_()
        # batch size = 64, 1 channel (grayscale), 224x224 image size, 26 classes, alexnet,
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 224 -> 224
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 224 -> 224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 112 -> 112
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 112 -> 112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 56 -> 56
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 56 -> 56
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 56 -> 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 28 -> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 28 -> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 28 -> 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14

            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 14 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 14 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 14 -> 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 -> 7
            # dense layers for vgg16
            nn.Linear(512 * 7 * 7, 4096),  # 25088 -> 4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)

        return x

    # TRAINING


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # save to file
            with open('train_loss.txt', 'a') as f:
                f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                f.write('\n')
                f.close()

import pytest
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target

@pytest.fixture
def model():
    return VGG16()
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_loader():

    x_test = np.random.rand(100, 1, 28, 28)
    y_test = np.random.randint(0, 26, 100)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = CustomDataset(x_test, y_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return test_loader
# TESTING
def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.10f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # save to file
    with open('test_loss.txt', 'a') as f:
        f.write('Test set: Average loss: {:.10f}, Accuracy: {}/{} ({:.0f}%);'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        f.write('\n')
        f.close()


# MAIN
class CustomDataset(Dataset):
    def _init_(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def _len_(self):
        return len(self.data)

    def _getitem_(self, index):
        x = self.data[index]
        y = self.labels[index]
        if self.transform:
            x = self.transform(x)
        return x, y


def main():
    print(torch.cuda.is_available())

    df = pd.read_csv('./A_Z Handwritten Data.csv')
    x = df.drop(df.columns[0], axis=1)
    y = df[df.columns[0]]
    x = np.array(x)
    y = np.array(y)
    # test,train,validation split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)
    x_val = x_val.reshape(-1, 1, 28, 28)

    # train
    train_dataset = CustomDataset(x_train, y_train, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.ToPILImage()

    ]))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # validation
    val_dataset = CustomDataset(x_val, y_val, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.ToPILImage()

    ]))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # test
    test_dataset = CustomDataset(x_test, y_test, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.ToPILImage()

    ]))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # load model
    model = VGG16()
    # set device to cpu
    device = torch.device('cpu')
    model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train the model
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)
        test_model(model, device, val_loader)  # updated function call

    # save the model
    torch.save(model.state_dict(), 'model.pt')

    # test the model
    test_model(model, device, test_loader)  # updated function call


if __name__ == "__main__":
    main()
