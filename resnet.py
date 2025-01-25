import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset


from sklearn.model_selection import train_test_split

import torch.optim as optim
import pandas as pd
import numpy as np
import json
import time

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

    
        

class ResNet(nn.Module):
    #batch size = 64, 1 channel (grayscale), 28x28 image size, 26 classes, resnet
    def __init__(self, num=26):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(BasicBlock(planes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    

    


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    
    def get_data(self):
        return self.data
    

def train_f(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0
            running_loss_hist = []
            # Iterate over data.
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                loss_float = loss.item()
                acc_float = (preds == labels.data).float().mean()
                running_loss_hist.append([loss_float, acc_float])

                


            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            with open('loss_past.txt', 'a') as f:
                #json
                json.dump(running_loss_hist, f)


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
        print()
    return train_loss, val_loss, train_acc, val_acc

def test_f(model, test_loader, criterion, optimizer, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

def val_f(model, val_loader, criterion, optimizer, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)

    print('Val Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc



def main():
    # Load the data
    data = pd.read_csv('./A_Z Handwritten Data.csv')
    # Split the data into training and testing
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    # Split the training data into training and validation
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    
    x_train = train.drop('0', axis=1).values
    y_train = train['0'].values
    x_val = val.drop('0', axis=1).values
    y_val = val['0'].values
    x_test = test.drop('0', axis=1).values
    y_test = test['0'].values


    x_train = x_train.reshape(-1,1,28,28)
    x_test = x_test.reshape(-1,1,28,28)
    x_val = x_val.reshape(-1,1,28,28)



    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    x_val = torch.from_numpy(x_val).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    y_val = torch.from_numpy(y_val).long()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),



    ])


    train_dataset = CustomDataset(x_train, y_train, transform=transform)
    test_dataset = CustomDataset(x_test, y_test, transform=transform)
    val_dataset = CustomDataset(x_val, y_val, transform=transform)

    #cuda
        
    print('loaded')
  
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, **kwargs)

    model = ResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loss, val_loss, train_acc, val_acc = train_f(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)
    test_loss, test_acc = test_f(model, test_loader, criterion, optimizer, device)
    val_loss, val_acc = val_f(model, val_loader, criterion, optimizer, device)

    # Save the model
    torch.save(model.state_dict(), 'model.pt')

    with open('loss.json', 'w') as f:
        json.dump({'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc, 'test_loss': test_loss, 'test_acc': test_acc}, f)
        
    
        



if __name__ == '__main__':
    main()

