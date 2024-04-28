import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from transforms import Transforms
from dataset import Facetrue_landmarksDataset
from model import Network

file = open('ibug_300W_large_face_landmark_dataset/helen/trainset/100032540_1.pts')
points = file.readlines()[3:-1]

true_landmarks = []

for point in points:
    x,y = point.split(' ')
    true_landmarks.append([floor(float(x)), floor(float(y[:-1]))])


dataset = Facetrue_landmarksDataset(Transforms())

# split the dataset into validation and test sets
len_valid_set = int(0.1*len(dataset))
len_train_set = len(dataset) - len_valid_set



train_dataset , valid_dataset,  = torch.utils.data.random_split(dataset , [len_train_set, len_valid_set])

# shuffle and batch the datasets
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4)


images, true_landmarks = next(iter(trainloader))

print(images.shape)
print(true_landmarks.shape)

import sys

def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))

    sys.stdout.flush()


torch.autograd.set_detect_anomaly(True)
network = Network()
network.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.0001)

loss_min = np.inf
num_epochs = 10

def train(num_epochs, trainloader, valloader, criterion, optimizer, network):
    for epoch in range(1,num_epochs+1):

        loss_train = 0
        loss_valid = 0
        running_loss = 0

        network.train()
        for step in range(1,len(trainloader)+1):

            images, true_landmarks = next(iter(trainloader))

            images = images.cuda()
            true_landmarks = true_landmarks.view(true_landmarks.size(0),-1).cuda()

            predictions = network(images)

            # clear all the gradients before calculating them
            optimizer.zero_grad()

            # find the loss for the current step
            loss_train_step = criterion(predictions, true_landmarks)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train/step

            print_overwrite(step, len(trainloader), running_loss, 'train')

        network.eval()
        with torch.no_grad():

            for step in range(1,len(valloader)+1):

                images, true_landmarks = next(iter(valloader))

                images = images.cuda()
                true_landmarks = true_landmarks.view(true_landmarks.size(0),-1).cuda()

                predictions = network(images)

                # find the loss for the current step
                loss_valid_step = criterion(predictions, true_landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step

                print_overwrite(step, len(valloader), running_loss, 'valid')

        loss_train /= len(trainloader)
        loss_valid /= len(valloader)

        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')

        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), '/content/face_true_landmarks.pth')
            print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))

    print('Training Done')

train(num_epochs, trainloader, valloader, criterion, optimizer, network)