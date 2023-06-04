import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from model import CNN
# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Hyper parameters
num_epochs = 20
batchsize = 4
lr = 0.001

EPOCHS = 2
BATCH_SIZE = 5
LEARNING_RATE = 0.003
TRAIN_DATA_PATH = "train/"
TEST_DATA_PATH = "test/"
TRANSFORM_IMG = transforms.Compose(
    [transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) 

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batchsize,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batchsize,
                                          shuffle=False)

classes = ('ripe', 'unripe')
cnn = CNN()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %
                  (epoch+1,num_epochs,i+1,len(train_data)//batchsize,loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var)
correct = 0
total = 0
for images, labels in test_loader:
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    print(predicted)
dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
print('Test Accuracy of the model on test images: %.6f%%' % (100.0*correct/total))

#Save the Trained Model
torch.save(cnn.state_dict(),'cnn.pkl')