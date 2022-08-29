############################################################## Image Classification ##############################################################

#You'll be required to create your own custom model using nn.Module following an architecture described in the assignment. You have been given a code template following which you'll be able to train your model on the MNIST dataset.
#MNIST is a dataset of 60000 grayscale training images and 10000 grayscale test images with each of their resolution being 28 x 28.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

# Data
print('Data transformation')
transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

trainset = torchvision.datasets.MNIST(
    root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(
    root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2)

#Visualize the image from training dataset
images, labels = next(iter(trainloader))
plt.imshow(images[43].reshape(28,28), cmap="gray")
print(labels[30]) #image index

#Model
print('Model creation')
# Create the model as described in question
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


    def forward(self, x):


net = net.to(device)
#Define optimizer,loss function,learning rate scheduler

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Write your training code here

    
    


# Testing
def test(epoch):
    global best_acc
    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            

    # Save checkpoint for the model which yields best accuracy
    

for epoch in range(0,100):
    print("Training")
    train(epoch)
    print("Testing")
    test(epoch)
