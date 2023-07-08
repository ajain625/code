import torch
import torch.nn as nn
import torchvision
import numpy as np

import utils
import models

net1 = models.LeNet5(10)
net2 = models.LeNet5(10)

#print(net1.fc2.weight.data.shape)
#print(net2.fc1.weight.data.shape)
# print(net1.state_dict()['layer1.0.bias'].shape)
# print(net1.state_dict()['layer1.1.bias'].shape)
# print(net1.state_dict()['layer1.0.weight'].shape)
# print(net1.state_dict()['layer1.1.weight'].shape)

optim = torch.optim.SGD([{'params': net1.parameters()},{'params': net2.parameters()}], lr=10)
loss_fn = nn.MSELoss()

print(((net1.fc1.weight.data-net2.fc1.weight.data)*2).sum())
print(((net1.fc2.weight.data-net2.fc2.weight.data)*2).sum())
print("training")
for epoch in range(100):
    #loss = torch.tensor([0.0], requires_grad=True)
    #penalty = torch.tensor(((net1.fc2.weight.data - net2.fc2.weight.data)**2).sum(), requires_grad=True)
    loss = loss_fn(net1.fc2.weight, net2.fc2.weight)
    loss.backward()
    print(net1.fc1.weight.grad)
    optim.step()
    print(loss.item())
    optim.zero_grad()
print("done")
print(((net1.fc1.weight.data-net2.fc1.weight.data)*2).sum())
print(((net1.fc2.weight.data-net2.fc2.weight.data)*2).sum())
