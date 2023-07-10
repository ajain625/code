import torch
import torch.nn as nn
import torchvision
import numpy as np

import utils
import models

# net1 = models.LeNet5(10)
# net2 = models.LeNet5(10)

# print(net1.state_dict()['layer1.0.bias'].shape)
# print(net1.state_dict()['layer1.1.bias'].shape)
# print(net1.state_dict()['layer1.0.weight'].shape)
# print(net1.state_dict()['layer1.1.weight'].shape)

# optim = torch.optim.SGD([{'params': net1.parameters()},{'params': net2.parameters()}], lr=10)
# loss_fn = nn.MSELoss()

# print(((net1.fc1.weight.data-net2.fc1.weight.data)*2).sum())
# print(((net1.fc2.weight.data-net2.fc2.weight.data)*2).sum())
# print("training")
# for epoch in range(100):
#     #loss = torch.tensor([0.0], requires_grad=True)
#     #penalty = torch.tensor(((net1.fc2.weight.data - net2.fc2.weight.data)**2).sum(), requires_grad=True)
#     loss = loss_fn(net1.fc2.weight, net2.fc2.weight)
#     loss.backward()
#     print(net1.fc1.weight.grad)
#     optim.step()
#     print(loss.item())
#     optim.zero_grad()
# print("done")
# print(((net1.fc1.weight.data-net2.fc1.weight.data)*2).sum())
# print(((net1.fc2.weight.data-net2.fc2.weight.data)*2).sum())

def lenet_coupling_alt(checkpoint1, checkpoint2, class1a, class1b, class2a, class2b, save_path = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/coupled', epochs=50, batch_size = 128, lr=0.01, momentum=0.9, weight_decay=0.0001, coupling_weight = 1, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.get_device_name())

    net1 = models.LeNet5(2)
    net2 = models.LeNet5(2)
    net1.load_state_dict(torch.load(checkpoint1)['model_state_dict'])
    net2.load_state_dict(torch.load(checkpoint2)['model_state_dict'])
    net1.to(device)
    net2.to(device)
    print('models loaded')
    
    optimizer = torch.optim.SGD([{'params': net1.parameters()},{'params': net2.parameters()}], lr=lr, momentum=momentum, weight_decay=weight_decay)
    train_data_a, train_labels_a, test_data_a, test_labels_a = utils.load_cifar100_2_classes(class1a, class2a, gray=True, seed=seed)
    train_data_b, train_labels_b, test_data_b, test_labels_b = utils.load_cifar100_2_classes(class1b, class2b, gray=True, seed=seed)
    assert train_data_a.shape[0] == train_data_b.shape[0]
    cross_entropy_loss = nn.CrossEntropyLoss()
    coupling_loss = nn.MSELoss()
    batches = int(train_data_a.shape[0]/batch_size)
    print('training starting')

    for epoch in range(epochs):
        for batch in range(batches-1):
            images_a = torch.from_numpy(train_data_a[batch*batch_size:(batch+1)*batch_size]).to(device)
            labels_a = torch.from_numpy(train_labels_a[batch*batch_size:(batch+1)*batch_size]).to(device)
            outputs_a = net1(images_a)
            images_b = torch.from_numpy(train_data_b[batch*batch_size:(batch+1)*batch_size]).to(device)
            labels_b = torch.from_numpy(train_labels_b[batch*batch_size:(batch+1)*batch_size]).to(device)
            outputs_b = net2(images_b)
            loss = cross_entropy_loss(outputs_a, labels_a) + cross_entropy_loss(outputs_b, labels_b) + coupling_weight*coupling_loss(net1.fc2.weight, net2.fc2.weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, epochs, batch+1, batches, loss.item(), utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())))
            del images_a, labels_a, outputs_a, images_b, labels_b, outputs_b
            torch.cuda.empty_cache()
        images_a = torch.from_numpy(train_data_a[(batch+1)*batch_size:]).to(device)
        labels_a = torch.from_numpy(train_labels_a[(batch+1)*batch_size:]).to(device)
        outputs_a = net1(images_a)
        images_b = torch.from_numpy(train_data_b[(batch+1)*batch_size:]).to(device)
        labels_b = torch.from_numpy(train_labels_b[(batch+1)*batch_size:]).to(device)
        outputs_b = net2(images_b)
        loss = cross_entropy_loss(outputs_a, labels_a) + cross_entropy_loss(outputs_b, labels_b) + coupling_weight*coupling_loss(net1.fc2.weight, net2.fc2.weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, epochs, batch+1, batches, loss.item(), utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())))
        del images_a, labels_a, outputs_a, images_b, labels_b, outputs_b
        torch.cuda.empty_cache()

        # Test Set	
        with torch.no_grad():
            net1.eval()
            net2.eval()
            images_a = torch.from_numpy(test_data_a).to(device)
            labels_a = torch.from_numpy(test_labels_a).to(device)
            outputs_a = net1(images_a)
            images_b = torch.from_numpy(test_data_b).to(device)
            labels_b = torch.from_numpy(test_labels_b).to(device)
            outputs_b = net2(images_b)
            test_accuracy_a = utils.cifar_accuracy(labels_a.detach().cpu().numpy(), outputs_a.detach().cpu().numpy())
            test_accuracy_b = utils.cifar_accuracy(labels_b.detach().cpu().numpy(), outputs_b.detach().cpu().numpy())
            print('(Test Set) Epoch: [{}/{}], Train Loss: {}, Accuracy Task A: {}, Accuracy Task B: {}'.format(epoch+1, epochs, loss.item(), test_accuracy_a, test_accuracy_b))
            del images_a, labels_a, outputs_a, images_b, labels_b, outputs_b
            torch.cuda.empty_cache()
        net1.train()
        net2.train()
        # Save Models
        torch.save({'epoch': epoch, 'model_state_dict': net1.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy_a}, save_path +'lenet_coupled_a'+str(class1a)+'vs'+str(class2a)+ '_coupling_' + str(coupling_weight)+'.pth')
        torch.save({'epoch': epoch, 'model_state_dict': net2.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy_b}, save_path +'lenet_coupled_b'+str(class1b)+'vs'+str(class2b)+ '_coupling_' + str(coupling_weight)+'.pth')
    return test_accuracy_a, test_accuracy_b

def lenet_coupling(checkpoint, class1a, class1b, class2a, class2b, save_path = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/coupled/', epochs=50, batch_size = 128, lr=0.01, momentum=0.9, weight_decay=0.0001, coupling_weight = 1, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.get_device_name())

    net1 = models.LeNet5(2)
    net2 = models.LeNet5(2)
    net1.load_state_dict(torch.load(checkpoint)['model_state_dict'])
    net2.load_state_dict(torch.load(checkpoint)['model_state_dict'])
    print(f'Accuracy A : {torch.load(checkpoint)["test_accuracy_a"]}, Accuracy B : {torch.load(checkpoint)["test_accuracy_b"]}')
    net1.to(device)
    net2.to(device)

    # Freeze all layers except last fc layer
    for param in net1.parameters():
        param.requires_grad = False
    for param in net2.parameters():
        param.requires_grad = False
    for param in net1.fc2.parameters():
        param.requires_grad = True
    for param in net2.fc2.parameters():
        param.requires_grad = True 
    print('models loaded')
    
    optimizer = torch.optim.SGD([{'params': net1.parameters()},{'params': net2.parameters()}], lr=lr, momentum=momentum, weight_decay=weight_decay)
    train_data_a, train_labels_a, test_data_a, test_labels_a = utils.load_cifar100_2_classes(class1a, class2a, gray=True, seed=seed)
    train_data_b, train_labels_b, test_data_b, test_labels_b = utils.load_cifar100_2_classes(class1b, class2b, gray=True, seed=seed)
    assert train_data_a.shape[0] == train_data_b.shape[0]
    cross_entropy_loss = nn.CrossEntropyLoss()
    coupling_loss = nn.MSELoss()
    batches = int(train_data_a.shape[0]/batch_size)
    print('training starting')

    for epoch in range(epochs):
        for batch in range(batches-1):
            images_a = torch.from_numpy(train_data_a[batch*batch_size:(batch+1)*batch_size]).to(device)
            labels_a = torch.from_numpy(train_labels_a[batch*batch_size:(batch+1)*batch_size]).to(device)
            outputs_a = net1(images_a)
            images_b = torch.from_numpy(train_data_b[batch*batch_size:(batch+1)*batch_size]).to(device)
            labels_b = torch.from_numpy(train_labels_b[batch*batch_size:(batch+1)*batch_size]).to(device)
            outputs_b = net2(images_b)
            loss = cross_entropy_loss(outputs_a, labels_a) + cross_entropy_loss(outputs_b, labels_b) + coupling_weight*coupling_loss(net1.fc2.weight, net2.fc2.weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, epochs, batch+1, batches, loss.item(), utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())))
            del images_a, labels_a, outputs_a, images_b, labels_b, outputs_b
            torch.cuda.empty_cache()
        images_a = torch.from_numpy(train_data_a[(batch+1)*batch_size:]).to(device)
        labels_a = torch.from_numpy(train_labels_a[(batch+1)*batch_size:]).to(device)
        outputs_a = net1(images_a)
        images_b = torch.from_numpy(train_data_b[(batch+1)*batch_size:]).to(device)
        labels_b = torch.from_numpy(train_labels_b[(batch+1)*batch_size:]).to(device)
        outputs_b = net2(images_b)
        loss = cross_entropy_loss(outputs_a, labels_a) + cross_entropy_loss(outputs_b, labels_b) + coupling_weight*coupling_loss(net1.fc2.weight, net2.fc2.weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, epochs, batch+1, batches, loss.item(), utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())))
        del images_a, labels_a, outputs_a, images_b, labels_b, outputs_b
        torch.cuda.empty_cache()

        # Test Set	
        with torch.no_grad():
            net1.eval()
            net2.eval()
            images_a = torch.from_numpy(test_data_a).to(device)
            labels_a = torch.from_numpy(test_labels_a).to(device)
            outputs_a = net1(images_a)
            images_b = torch.from_numpy(test_data_b).to(device)
            labels_b = torch.from_numpy(test_labels_b).to(device)
            outputs_b = net2(images_b)
            test_accuracy_a = utils.cifar_accuracy(labels_a.detach().cpu().numpy(), outputs_a.detach().cpu().numpy())
            test_accuracy_b = utils.cifar_accuracy(labels_b.detach().cpu().numpy(), outputs_b.detach().cpu().numpy())
            print('(Test Set) Epoch: [{}/{}], Train Loss: {}, Accuracy Task A: {}, Accuracy Task B: {}'.format(epoch+1, epochs, loss.item(), test_accuracy_a, test_accuracy_b))
            del images_a, labels_a, outputs_a, images_b, labels_b, outputs_b
            torch.cuda.empty_cache()
        net1.train()
        net2.train()
        # Save Models
        torch.save({'epoch': epoch, 'model_state_dict': net1.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy_a}, save_path +'lenet_coupled_'+str(class1a)+'vs'+str(class2a)+ '_coupling_' + str(coupling_weight)+'.pth')
        torch.save({'epoch': epoch, 'model_state_dict': net2.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy_b}, save_path +'lenet_coupled_'+str(class1b)+'vs'+str(class2b)+ '_coupling_' + str(coupling_weight)+'.pth')
    return test_accuracy_a, test_accuracy_b


if __name__ == '__main__':
    #lenet_coupling_alt(class1a=1, class2a=3, class1b=9, class2b=13, checkpoint1='/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/lenet_individual_1vs3.pth', checkpoint2='/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/lenet_individual_9vs13.pth', epochs=100, batch_size=128, coupling_weight=1)
    lenet_coupling(checkpoint = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/lenet_joint_1_9vs3_13.pth', class1a=1, class2a=3, class1b=9, class2b=13, epochs=100, batch_size=128, coupling_weight=1, lr=0.001)
