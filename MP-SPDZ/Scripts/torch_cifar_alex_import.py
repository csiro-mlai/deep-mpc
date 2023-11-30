#!/usr/bin/env python3

# test model output by torch_alex_test.mpc

import torchvision
import torch
import torch.nn as nn
import numpy

net = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(64, 96, kernel_size=3, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(96, 96, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(96, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(1024, 128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

f = open('Player-Data/Binary-Output-P0-0')

state = net.state_dict()

for name in state:
    shape = state[name].shape
    size = numpy.prod(shape)
    var = numpy.fromfile(f, 'double', count=size)
    var = var.reshape(shape)
    state[name] = torch.Tensor(var)

net.load_state_dict(state)

get_data = lambda train, transform=None: torchvision.datasets.CIFAR10(
    root='/tmp', train=train, download=True, transform=transform)

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), lambda x: 2 * x - 1])

with torch.no_grad():
    ds = get_data(False, transform)
    total = correct_classified = 0
    for data in torch.utils.data.DataLoader(ds, batch_size=128):
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_classified += (predicted == labels).sum().item()
    test_acc = (100 * correct_classified / total)
    print('Test accuracy of the network: %.2f %%' % test_acc)
