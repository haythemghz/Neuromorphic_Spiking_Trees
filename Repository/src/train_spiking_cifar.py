import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

class SpikingCNNFrontEnd(nn.Module):
    """
    A small learned CNN front-end designed to be frozen/used for spiking features.
    """
    def __init__(self):
        super(SpikingCNNFrontEnd, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x # Returns 32 x 8 x 8 features for CIFAR-10

def train_cifar_fe():
    print("Training Spiking CNN Front-End on CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    
    model = SpikingCNNFrontEnd()
    # For a front-end, we just need it to learn features. 
    # One hack is to add a dummy classifier and train for 2 epochs.
    classifier = nn.Sequential(model, nn.Flatten(), nn.Linear(32*8*8, 10))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    classifier.train()
    for epoch in range(1): # Quick feature learning
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i > 100: break # Early signal suffice for FE
    
    torch.save(model.state_dict(), 'spiking_cifar_fe.pth')
    print("Front-end saved to spiking_cifar_fe.pth")

if __name__ == "__main__":
    train_cifar_fe()
