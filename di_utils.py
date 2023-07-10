import torch
import torch.nn as nn
import torchvision
import numpy as np

import utils
import models


def cifar100_individual_train(model, class1, class2, save_path = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/', epochs=500, batch_size = 128, lr=0.01, momentum=0.9, weight_decay=0.0001, seed=42):
    assert model in ['lenet', 'resnet']
    assert class1 in range(1, 21)
    assert class2 in range(1, 21)
    assert class1 != class2	
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.get_device_name())
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model == 'lenet':
        net = models.LeNet5(2)
        train_data = utils.normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class1)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class2)+'.npy')]), gray=True)
        train_labels = np.hstack([np.zeros(len(train_data)//2, dtype=int), np.ones(len(train_data)//2, dtype=int)])
        test_data = utils.normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class1)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class2)+'.npy')]), gray=True)
        test_labels = np.hstack([np.zeros(len(test_data)//2, dtype=int), np.ones(len(test_data)//2, dtype=int)])
        print('data and lenet loaded')
    elif model == 'resnet':
        net = torchvision.models.resnet18()
        net.fc = nn.Linear(512, 2)
        train_data = utils.normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class1)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class2)+'.npy')]))
        train_labels = np.hstack([np.zeros(len(train_data)//2, dtype=int), np.ones(len(train_data)//2, dtype=int)])
        test_data = utils.normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class1)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class2)+'.npy')]))
        test_labels = np.hstack([np.zeros(len(test_data)//2, dtype=int), np.ones(len(test_data)//2, dtype=int)])
        print('data and resnet loaded')
    

    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    print(f'class1: {class1}, class2: {class2}')

    shuffle = np.random.permutation(len(train_data))
    train_data = train_data[shuffle]
    train_labels = train_labels[shuffle]

    batches = len(train_data)//batch_size

    for epoch in range(epochs):
        for batch in range(batches-1):
            images = torch.from_numpy(train_data[batch*batch_size:(batch+1)*batch_size]).to(device)
            labels = torch.from_numpy(train_labels[batch*batch_size:(batch+1)*batch_size]).to(device)
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, epochs, batch+1, batches, loss.item(), utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())))
            del images, labels, outputs
            torch.cuda.empty_cache()
        images = torch.from_numpy(train_data[(batches-1)*batch_size:]).to(device)
        labels = torch.from_numpy(train_labels[(batches-1)*batch_size:]).to(device)
        outputs = net(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, epochs, batches, batches, loss.item(), utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())))
        del images, labels, outputs
        torch.cuda.empty_cache()

        # Test Set	
        with torch.no_grad():
            net.eval()
            images = torch.from_numpy(test_data).to(device)
            labels = torch.from_numpy(test_labels).to(device)
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            test_accuracy = utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            print('(Test Set) Epoch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, epochs, loss.item(), test_accuracy))
            del images, labels, outputs
            torch.cuda.empty_cache()
        net.train()
        # Save Model
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy}, save_path + model+'_individual_'+str(class1)+'vs'+str(class2)+'.pth')
    return test_accuracy

def cifar100_joint_train(model, class1a, class1b, class2a, class2b, save_path = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/', epochs=500, batch_size = 128, lr=0.01, momentum=0.9, weight_decay=0.0001, seed=42):
    assert model in ['lenet', 'resnet']
    assert class1a in range(1, 21)
    assert class1b in range(1, 21)
    assert class2a in range(1, 21)
    assert class2b in range(1, 21)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.get_device_name())
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model == 'lenet':
        net = models.LeNet5(2)
        train_data = np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class1a)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class1b)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class2a)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class2b)+'.npy')]).mean(axis=1, keepdims=True)
        train_labels = np.hstack([np.zeros(len(train_data)//2, dtype=int), np.ones(len(train_data)//2, dtype=int)])
        test_data_a = np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class1a)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class2a)+'.npy')]).mean(axis=1, keepdims=True)
        test_data_b = np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class1b)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class2b)+'.npy')]).mean(axis=1, keepdims=True)
        test_labels = np.hstack([np.zeros(len(test_data_a)//2, dtype=int), np.ones(len(test_data_a)//2, dtype=int)])
        print('data and lenet loaded')
    elif model == 'resnet':
        net = torchvision.models.resnet18()
        net.fc = nn.Linear(512, 2)
        train_data = np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class1a)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class1b)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class2a)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class2b)+'.npy')])
        train_labels = np.hstack([np.zeros(len(train_data)//2, dtype=int), np.ones(len(train_data)//2, dtype=int)])
        test_data_a = np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class1a)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class2a)+'.npy')])
        test_data_b = np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class1b)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class2b)+'.npy')])
        test_labels = np.hstack([np.zeros(len(test_data_a)//2, dtype=int), np.ones(len(test_data_a)//2, dtype=int)])
        print('data and resnet loaded')
    

    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    print(f'Class 1a : {class1a} Class 1b {class1b} Class 2a {class2a} Class 2b {class2b}')

    shuffle = np.random.permutation(len(train_data))
    train_data = train_data[shuffle]
    train_labels = train_labels[shuffle]

    batches = len(train_data)//batch_size

    for epoch in range(epochs):
        for batch in range(batches-1):
            images = torch.from_numpy(train_data[batch*batch_size:(batch+1)*batch_size]).to(device)
            labels = torch.from_numpy(train_labels[batch*batch_size:(batch+1)*batch_size]).to(device)
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, epochs, batch+1, batches, loss.item(), utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())))
            del images, labels, outputs
            torch.cuda.empty_cache()
        images = torch.from_numpy(train_data[(batches-1)*batch_size:]).to(device)
        labels = torch.from_numpy(train_labels[(batches-1)*batch_size:]).to(device)
        outputs = net(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, epochs, batches, batches, loss.item(), utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())))
        del images, labels, outputs
        torch.cuda.empty_cache()

        # Test Set	
        with torch.no_grad():
            net.eval()
            images_a = torch.from_numpy(test_data_a).to(device)
            images_b = torch.from_numpy(test_data_b).to(device)
            labels = torch.from_numpy(test_labels).to(device)
            outputs_a = net(images_a)
            outputs_b = net(images_b)
            test_accuracy_a = utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs_a.detach().cpu().numpy())
            test_accuracy_b = utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs_b.detach().cpu().numpy())
            print('(Test Set) Epoch: [{}/{}], Accuracy Task A : {}, Accuracy Task B: {}'.format(epoch+1, epochs, test_accuracy_a, test_accuracy_b))
            del images_a, images_b, labels, outputs_a, outputs_b
            torch.cuda.empty_cache()
        net.train()
        # Save Model
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy_a': test_accuracy_a, 'test_accuracy_b':test_accuracy_b}, save_path + model+'_joint_'+str(class1a) + '_' + str(class1b)+'vs'+ str(class2a) + '_' + str(class2b) +'.pth')
    return test_accuracy_a, test_accuracy_b

# aquatic - 1, flower - 3
# medium mammals - 13, large carnivores - 9
cifar100_joint_train('resnet', 1, 9, 3, 13)
cifar100_joint_train('lenet', 1, 9, 3, 13)
cifar100_individual_train('resnet', 1, 3)
cifar100_individual_train('lenet', 1, 3)
cifar100_individual_train('resnet', 9, 13)
cifar100_individual_train('lenet', 9, 13)