import torch
from torch import nn 
from torchvision import datasets, transforms


# prepocesing image

transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])


batch_size=64 
trainset = datasets.FashionMNIST('Fashion_MNIST/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('Fashion_MNIST/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
print("hello")

class FashionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1=nn.Linear(784,256)
        self.hidden2=nn.Linear(256,128)
        self.output=nn.Linear(128,10)
        self.softmax=nn.Softmax(dim=1)
        self.activation=nn.ReLU()
    def forward(self,x):
        x = self.hidden1(x)
        x=self.activation(x)
        x=self.hidden2(x)
        x=self.activation(x)
        x=self.output(x)
        output=self.softmax(x)
        return(output)



model=FashionNetwork()
print(model)
criterion=nn.NLLLoss()


from torch import optim
optimizer=optim.Adam(model.parameters())
optimizer.defaults

optimizer=optim.Adam(model.parameters(),lr=3e-3)
optimizer.defaults
epoch=10

for _ in range(epoch):
    running_loss=0
    for image, label in trainloader:
        optimizer.zero_grad()
        image=image.view(image.shape[0],-1)
        pred=model(image)
        loss=criterion(pred,label)
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()
    else:
        print(f'Training loss: {running_loss/len(trainloader):.4f}')

torch.tensor([[1]]).item()




class FashionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)
        self.log_softmax = nn.LogSoftmax()
        self.activation = nn.ReLU()
        self.drop = nn.Dropout(p=0.25)
    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.output(x)
        output = self.log_softmax(x)
        return output



model = FashionNetwork()
model

import torch.nn.functional as F


class FashionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784,256)
        self.hidden2 = nn.Linear(256,128)
        self.output = nn.Linear(128,10)
        
        
    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.log_softmax(self.output(x))
        return x












