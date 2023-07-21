import torch
import torch.nn as nn
import numpy as np

import utils
import models


def cifar100_individual_train(model, class1, class2, save_path = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/coarse', epochs=500, batch_size = 128, lr=0.01, momentum=0.9, weight_decay=0.0001, random_split=True, seed=42, split_order = None, fine=False, reduction=1):
    assert model in ['lenet', 'resnet']
    #assert class1 in range(1, 21)
    #assert class2 in range(1, 21)
    assert class1 != class2	
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    #assert device == 'cuda'
    print(torch.cuda.get_device_name())
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model == 'lenet':
        net = models.LeNet5(2)
        train_data, train_labels, test_data, test_labels = utils.load_cifar100_2_classes(class1, class2, gray=True, shuffle=True, split_order=split_order, seed=seed, fine=fine, random_split=random_split, reduction=reduction)
        print('data and lenet loaded')
    elif model == 'resnet':
        #net = torchvision.models.resnet18()
        net = models.resnet18vw(width=32)
        net.fc = nn.Linear(256, 2)
        train_data, train_labels, test_data, test_labels = utils.load_cifar100_2_classes(class1, class2, gray=False, shuffle=True, split_order=split_order, seed=seed, fine=fine, random_split=random_split, reduction=reduction)
        print('data and resnet loaded')

    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    batches = len(train_data)//batch_size
    print(f'class1: {class1}, class2: {class2}')
    seeds = np.arange(epochs)

    for epoch in range(epochs):
        np.random.seed(seeds[epoch])
        shuffle = np.random.permutation(len(train_data))
        train_data = train_data[shuffle]
        train_labels = train_labels[shuffle]
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
            if epoch%10 == 0:
                print('(Test Set) Epoch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, epochs, loss.item(), test_accuracy))
            del images, labels, outputs
            torch.cuda.empty_cache()
        net.train()
        # Save Model
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy}, save_path + model+'_individual_'+str(class1)+'vs'+str(class2)+'.pth')
    return test_accuracy

def cifar100_joint_train(model, class1a, class1b, class2a, class2b, save_path = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/coarse', epochs=500, batch_size = 128, lr=0.01, momentum=0.9, weight_decay=0.0001, random_split=True, seed=42, split_order = None, fine=False, reduction=1):
    assert model in ['lenet', 'resnet']
    #assert class1a in range(1, 21)
    #assert class1b in range(1, 21)
    #assert class2a in range(1, 21)
    #assert class2b in range(1, 21)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #assert device == 'cuda'
    print(torch.cuda.get_device_name())
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model == 'lenet':
        net = models.LeNet5(2)
        train_data, train_labels, test_data_a, test_data_b, test_labels = utils.load_cifar100_4_classes(class1a, class1b, class2a, class2b, gray=True, shuffle=True, random_split=random_split, split_order=split_order, seed=seed, fine=fine, reduction=reduction)
        print('data and lenet loaded')
    elif model == 'resnet':
        net = models.resnet18vw(width=32)
        net.fc = nn.Linear(256, 2)
        train_data, train_labels, test_data_a, test_data_b, test_labels = utils.load_cifar100_4_classes(class1a, class1b, class2a, class2b, gray=False, shuffle=True, random_split=random_split, split_order=split_order, seed=seed, fine=fine, reduction=reduction)
        print('data and resnet loaded')

    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    batches = len(train_data)//batch_size
    print(f'Class 1a : {class1a} Class 1b : {class1b} Class 2a : {class2a} Class 2b : {class2b}')
    seeds = np.arange(epochs)

    for epoch in range(epochs):
        np.random.seed(seeds[epoch])
        shuffle = np.random.permutation(len(train_data))
        train_data = train_data[shuffle]
        train_labels = train_labels[shuffle]
        del shuffle
        torch.cuda.empty_cache()
        for batch in range(batches):
            if batch != batches-1:
                images = torch.from_numpy(train_data[batch*batch_size:(batch+1)*batch_size]).to(device)
                labels = torch.from_numpy(train_labels[batch*batch_size:(batch+1)*batch_size]).to(device)
            else:
                images = torch.from_numpy(train_data[batch*batch_size:]).to(device)
                labels = torch.from_numpy(train_labels[batch*batch_size:]).to(device)
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {}, Accuracy: {}'.format(epoch+1, epochs, batch+1, batches, loss.item(), utils.cifar_accuracy(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())))
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
            if epoch%10 == 0:
                print('(Test Set) Epoch: [{}/{}], Accuracy Task A : {}, Accuracy Task B: {}'.format(epoch+1, epochs, test_accuracy_a, test_accuracy_b))
            del images_a, images_b, labels, outputs_a, outputs_b
            torch.cuda.empty_cache()
        net.train()
        # Save Model
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy_a': test_accuracy_a, 'test_accuracy_b':test_accuracy_b}, save_path + model+'_joint_'+str(class1a) + '_' + str(class1b)+'vs'+ str(class2a) + '_' + str(class2b) +'.pth')
    return test_accuracy_a, test_accuracy_b

def disparate_impact(model, class1a, class1b, class2a, class2b, cross=False, seed=42, save_path = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/coarse', epochs=500, batch_size = 128, lr=0.01, momentum=0.9, weight_decay=0.0001, fine=False, random_split=True):
    print('Disparate Impact Analysis')
    print('Model: ', model)
    print('Classes: ', class1a, class1b, class2a, class2b)
    assert model in ['lenet', 'resnet']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if fine:
        split_order = np.random.permutation(600)
    else:
        split_order = np.random.permutation(3000)
    joint_accuracy_a, joint_accuracy_b = cifar100_joint_train(model, class1a, class1b, class2a, class2b, seed=seed, save_path=save_path, epochs=epochs, batch_size=batch_size, lr=lr, momentum=momentum, weight_decay=weight_decay, split_order=split_order, fine=fine, random_split=random_split)
    individual_accuracy_a = cifar100_individual_train(model, class1a, class2a, seed=seed, save_path=save_path, epochs=epochs, batch_size=batch_size, lr=lr, momentum=momentum, weight_decay=weight_decay, split_order=split_order, fine=fine, random_split=random_split)
    individual_accuracy_b = cifar100_individual_train(model, class1b, class2b, seed=seed, save_path=save_path, epochs=epochs, batch_size=batch_size, lr=lr, momentum=momentum, weight_decay=weight_decay, split_order=split_order, fine=fine, random_split=random_split)
    disparate_impact = (joint_accuracy_a - joint_accuracy_b)/(individual_accuracy_a - individual_accuracy_b)
    if cross:
        cross_accuracy_a = cifar100_individual_train(model, class1a, class2b, seed=seed, save_path=save_path, epochs=epochs, batch_size=batch_size, lr=lr, momentum=momentum, weight_decay=weight_decay, split_order=split_order, fine=fine, random_split=random_split)
        cross_accuracy_b = cifar100_individual_train(model, class1b, class2a, seed=seed, save_path=save_path, epochs=epochs, batch_size=batch_size, lr=lr, momentum=momentum, weight_decay=weight_decay, split_order=split_order, fine=fine, random_split=random_split)
        print('Disparate Impact: {}, Individual Accuracy A: {}, Individual Accuracy B: {}, Joint Accuracy A: {}, Joint Accuracy B: {}, Cross Accuracy A: {}, Cross Accuracy B: {}'.format(disparate_impact, individual_accuracy_a, individual_accuracy_b, joint_accuracy_a, joint_accuracy_b, cross_accuracy_a, cross_accuracy_b))
        return disparate_impact, individual_accuracy_a, individual_accuracy_b, joint_accuracy_a, joint_accuracy_b, cross_accuracy_a, cross_accuracy_b
    else:
        print('Disparate Impact: {}, Individual Accuracy A: {}, Individual Accuracy B: {}, Joint Accuracy A: {}, Joint Accuracy B: {}'.format(disparate_impact, individual_accuracy_a, individual_accuracy_b, joint_accuracy_a, joint_accuracy_b))
        return disparate_impact, individual_accuracy_a, individual_accuracy_b, joint_accuracy_a, joint_accuracy_b
    
if __name__ == '__main__':
    # aquatic - 1, flower - 3
    # medium mammals - 13, large carnivores - 9
    #cifar100_joint_train('resnet', 1, 9, 3, 13)
    #cifar100_joint_train('lenet', 1, 9, 3, 13)
    #cifar100_individual_train('resnet', 1, 3)
    #cifar100_individual_train('lenet', 1, 3)
    #cifar100_individual_train('resnet', 9, 13)
    #cifar100_individual_train('lenet', 9, 13)
    #disparate_impact('resnet', class1a = 1, class1b = 9, class2a = 3, class2b = 13, cross=True, seed=42)
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tasks = np.array([[ 4,  1,  9,  8],
       [ 8,  5,  4,  3],
       [19, 14,  2,  1],
       [ 3,  7,  8, 17],
       [20,  1, 18,  7],
       [18, 14,  8, 15],
       [19,  9,  1,  6],
       [14, 11,  9,  5],
       [ 7, 11,  4,  3],
       [13,  4, 12, 18],
       [20,  9,  2, 15],
       [18,  4, 13,  3],
       [18, 10, 12,  7],
       [ 3,  2,  8, 10],
       [ 3,  8,  4, 13],
       [ 9, 15, 12,  6],
       [12, 20,  7,  9],
       [ 3,  6, 18,  8],
       [ 6, 15, 13,  9],
       [18,  8, 11,  2],
       [ 8,  2, 11, 13],
       [ 9,  3,  7, 11],
       [ 7, 16, 13, 15],
       [ 5,  9, 20,  8],
       [18, 20,  9, 14],
       [19, 13, 12,  8],
       [ 5, 17, 16,  3],
       [ 2,  4,  5,  6],
       [14,  3, 13, 18],
       [20, 15, 17,  9]])
    results = []
    for i in range(len(tasks)):
        print('Task: ', i)
        di, individual_accuracy_a, individual_accuracy_b, joint_accuracy_a, joint_accuracy_b, cross_accuracy_a, cross_accuracy_b = disparate_impact('resnet', class1a = tasks[i][0], class1b = tasks[i][1], class2a = tasks[i][2], class2b = tasks[i][3], cross=True, seed=seed, fine=False, save_path='/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/coarse/', random_split=True)
        results.append([tasks[i][0], tasks[i][1], tasks[i][2], tasks[i][3], di, individual_accuracy_a, individual_accuracy_b, joint_accuracy_a, joint_accuracy_b, cross_accuracy_a, cross_accuracy_b])
        np.savetxt('/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/coarse/resnet_30tasks_results.csv', np.array(results), delimiter=',', fmt='%f')