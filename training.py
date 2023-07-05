import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

import utils
import models

# Paths
save_path = '/nfs/ghome/live/ajain/checkpoints/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = models.LeNet5(10).to(device)

# Hyperparameters
num_epochs = 30
lr = 0.01
num_batches = 1

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_metric = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in range(num_batches):
        images, labels = utils.load_cifar_batch(batch+1, normalize=True)
        images = torch.from_numpy(images[:100, :, :, :]).to(device)
        labels = torch.from_numpy(labels[:100]).to(device)
        outputs = model(images)
        #print(outputs[:2])
        #print(labels[:2])
        loss = loss_metric(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, num_epochs, batch+1, num_batches, loss.item(), utils.cifar_accuracy(labels.detach().numpy(), outputs.detach().numpy())))
    # Test Set	
    model.eval()
    images, labels = utils.load_cifar_batch('test', normalize=True)
    images = torch.from_numpy(images).to(device)
    labels = torch.from_numpy(labels).to(device)
    outputs = model(images)
    loss = loss_metric(outputs, labels)
    #print('(Test Set) Epoch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, num_epochs, loss.item(), utils.cifar_accuracy(labels.detach().numpy(), outputs.detach().numpy())))
    model.train()
    # Save Model
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, save_path+'lenet_cifar10_'+str(epoch)+'.pth')