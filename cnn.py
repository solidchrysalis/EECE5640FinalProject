import torch
import sys
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import numpy as np
from optim_base import StochasticOptimizer, AdamOptimizer, AdagradOptimizer


dataset=sys.argv[1]
optim_name=sys.argv[2]
#dataset="cifar"
#optim_name="adagrad"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

epochs = 5

transform_cifar = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_fashion_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1 channel to 3 channels (RGB)
])

def get_data_loaders(dataset_name='cifar', batch_size=64, data_dir='./data'):
    if dataset_name.lower() == 'cifar':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_cifar
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_cifar
        )
    elif dataset_name.lower() == 'fashionmnist':
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform_fashion_mnist
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform_fashion_mnist
        )
    else:
        raise ValueError("Dataset name must be either 'cifar' or 'fashionmnist'")

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

trainloader, testloader = get_data_loaders(dataset_name='cifar', batch_size=64)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
# Custom optimizer class - TODO: Make Base Optimizer class
optimizer = None
if optim_name == "stochastic":
    optimizer = StochasticOptimizer.StochasticOptimizer(net.parameters(), lr=0.005)
elif optim_name == "adam":
    optimizer = AdamOptimizer.AdamOptimizer(net.parameters(), device, lr=0.005)
elif optim_name == "adagrad":
    optimizer = AdagradOptimizer.AdagradOptimizer(net.parameters(), device, lr=0.005)
else:
    print("Incorrect optimizer name")

print("running model")

for epoch in range(epochs): 

    running_loss = 0.0
    i = 0

    averages = []

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        start = time.monotonic_ns()
        optimizer.step()
        stop = time.monotonic_ns()

        averages.append(stop - start)

        running_loss += loss.item()
        if i % 500 == 499: 
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        i += 1
    

    print(f"Time: {(np.mean(averages)) / 1000000}")

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

