import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

REBUILD_DATA = True
IMG_SIZE = 50

class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)   
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)

net = Net()

class DogsVCats():
    CATS = 'PetImages/Cat' 
    DOGS = 'PetImages/Dog'
    LABELS = {CATS: 0, DOGS:1}
    training_data = []
    catcount = 0
    dogcount = 0 # always make it balanced

    def make_training_data(self ):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)): # progress bar
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, self.IMG_SIZE, self.IMG_SIZE) 
                    # convert to 1-hot vector (cats = 0, dogs = 1)
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]]) # see below 

                    if label == self .CATS:
                        self.catcount += 1
                    else:
                        self.dogcount += 1
                except Exception as e:
                    pass
                    print(str(e))

        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)
        print('Cats:', self.catcount)
        print('Dogs:', self.dogcount)

if REBUILD_DATA:
    dogsvcats = DogsVCats()
    dogsvcats.make_training_data()

training_data = np.load('training_data.py')
# np.eye(5) makes a matrix of size 5 with 1 on the diagonal
# np.eye(10)[7] to get a vector with all zeros and one at index 7