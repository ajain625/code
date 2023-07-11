import torch
import torch.nn as nn
import torchvision
import numpy as np

import utils
import models

# Paths
save_path = '/nfs/ghome/live/ajain/checkpoints/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name())
#model = models.LeNet5(10).to(device)
model = torchvision.models.resnet18().to(device)
model.fc = nn.Linear(512, 10).to(device)

# Hyperparameters
num_epochs = 100
lr = 0.003
num_batches = 5

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_metric = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in range(num_batches):
        images, labels = utils.load_cifar_batch(batch+1, normalize=True, gray=False)
        images = torch.from_numpy(images).to(device)
        labels = torch.from_numpy(labels).to(device)
        outputs = model(images)
        loss = loss_metric(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, num_epochs, batch+1, num_batches, loss.item(), utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())))
        del images, labels, outputs
        torch.cuda.empty_cache()
    # Test Set	
    with torch.no_grad():
        model.eval()
        images, labels = utils.load_cifar_batch('test', normalize=True, gray=False)
        images = torch.from_numpy(images).to(device)
        labels = torch.from_numpy(labels).to(device)
        outputs = model(images)
        loss = loss_metric(outputs, labels)
        print('(Test Set) Epoch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, num_epochs, loss.item(), utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())))
        del images, labels, outputs
        torch.cuda.empty_cache()
    model.train()
    # Save Model
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, save_path+'lenet_cifar10_'+str(epoch)+'.pth')