import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt  # pip install matplotlib

train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([
    transforms.ToTensor()
]))
test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([
    transforms.ToTensor()
]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


total = 0
counter_dict = {i:0 for i in range(10)}
for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1

for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total*100.0:.2f}%")

class Net(nn.Module):
    def __init__(self):
        super().__init__() # inheritance from the super class
        self.fc1 = nn.Linear(784, 64) # input size flattened 28x28 image
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)    

    def forward(self, x): # x is the data, but it has to scale properly
        x = F.relu(self.fc1(x)) # activation function to check if the neuron is firing or not (kinda stepped function, better use a sigmoid)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) # we don't want relu here, but only one should be fired, not part of them or multiple choices
        return F.log_softmax(x, dim=1) # always a flat layer, dim=1 is similar to axis in numpy/pandas

net = Net()
print(net)
X = torch.rand((28,28)) # pass random data into the NN
X = X.view(-1, 28*28)
output = net(X)
output

optimizer = optim.Adam(net.parameters(), lr=0.001) # takes everything taht is optimizable, first weights are static

EPOCHS = 3 # full pass through all data
for epoch in range(EPOCHS):
    for data in trainset: 
        # data is a batch of featuresets and labels
        X, y = data # unpack the tuple
        net.zero_grad() # we want to zero the gradient when training a new batch
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y) 
        # 1-hot vectors (check difference bet) and mean squared error
        # if output is just a single value, use nll_loss
        loss.backward()
        optimizer.step() # adjust weights 
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print('Accuracy: ', round(correct/total,3))
