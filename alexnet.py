import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset


from sklearn.model_selection import train_test_split

import torch.optim as optim
import pandas as pd
import numpy as np



import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True) 

#ALEXNET
class AlexNet(nn.Module):   
    def __init__(self, num=26):
        #Train Epoch: 1 [0/297960 (0%)]  Loss: 0.044079
        #Train Epoch: 1 [640/297960 (0%)]        Loss: nan


        super(AlexNet, self).__init__()
        #batch size = 64, 1 channel (grayscale), 224x224 image size, 26 classes, alexnet, 

        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2), #224 -> 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #55 -> 27
            nn.Conv2d(64, 192, kernel_size=5, padding=2), #27 -> 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #27 -> 13
            nn.Conv2d(192, 384, kernel_size=3, padding=1), #13 -> 13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), #13 -> 13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #13 -> 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #13 -> 6
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), #9216 -> 4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num),
        )


     
    
    def forward(self, x):

        x = self.feature(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x
    
    
    
#TRAINING
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            #save to file
            with open('loss.txt', 'a') as f:
                f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f};'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                f.write('\n')
                f.close()


#TESTING
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    #save to file
    with open('loss.txt', 'a') as f:
        f.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%);'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        f.write('\n')
        f.close()
   

    
#MAIN
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

def main():
    print(torch.cuda.is_available())

    df = pd.read_csv('./A_Z Handwritten Data.csv')
    x = df.drop(df.columns[0],axis=1)
    y = df[df.columns[0]]
    x = np.array(x)
    y = np.array(y)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,stratify = y,random_state = 42)

    x_train = x_train.reshape(-1,1,28,28)
    x_test = x_test.reshape(-1,1,28,28)



    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    alex_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))


    ])
    test_data = CustomDataset(x_test,y_test, transform=alex_transform)
    train_data = CustomDataset(x_train,y_train, transform=alex_transform)



    
    print('loaded')
  
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    #CUSTOM DATASET
    train_loader = DataLoader( 
        dataset = train_data,
        batch_size = 64,
        shuffle = True,
        **kwargs
    )
    test_loader = DataLoader(
        dataset = test_data,
        batch_size = 64,
        shuffle = True,
        **kwargs
    )

        

    model = AlexNet(num=26
                    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)




    for epoch in range(1, 10 + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


    torch.save(model.state_dict(),"mnist_cnn_epoch10.pth")
    

    
if __name__ == '__main__':
    main()
