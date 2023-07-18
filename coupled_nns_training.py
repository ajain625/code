import torch
import torch.nn as nn
import torchvision
import numpy as np

import utils
import models
import di_utils

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
    torch.cuda.empty_cache()
    torch.manual_seed(seed)

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
        torch.save({'epoch': epoch, 'model_state_dict': net1.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy_a}, save_path +'lenet_coupled_alt_'+str(class1a)+'vs'+str(class2a)+ '_coupling_' + str(coupling_weight)+'.pth')
        torch.save({'epoch': epoch, 'model_state_dict': net2.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy_b}, save_path +'lenet_coupled_alt_'+str(class1b)+'vs'+str(class2b)+ '_coupling_' + str(coupling_weight)+'.pth')
    return test_accuracy_a, test_accuracy_b

def lenet_coupling(checkpoint, class1a, class1b, class2a, class2b, save_path = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/coupled/', epochs=200, batch_size = 128, lr=0.01, momentum=0, weight_decay=0, coupling_weight = 1, random_split= False, seed=42, split_order = None, fine=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.get_device_name())
    torch.cuda.empty_cache()
    torch.manual_seed(seed)

    net1 = models.LeNet5(2)
    net2 = models.LeNet5(2)
    checkpoint1 = torch.load(checkpoint)['model_state_dict']
    checkpoint2 = torch.load(checkpoint)['model_state_dict']
    #checkpoint1 = {k:v for k,v in checkpoint1.items() if not k.startswith('fc')}
    #checkpoint2 = {k:v for k,v in checkpoint2.items() if not k.startswith('fc')}
    net1.load_state_dict(torch.load(checkpoint)['model_state_dict'], strict=True)
    net2.load_state_dict(torch.load(checkpoint)['model_state_dict'], strict=True)
    print(f'Accuracy A : {torch.load(checkpoint)["test_accuracy_a"]}, Accuracy B : {torch.load(checkpoint)["test_accuracy_b"]}')
    net1.to(device)
    net2.to(device)

    best_accuracy = 0
    best_accuracy_a = 0
    best_accuracy_b = 0
    best_epoch = 0

    # Freeze conv layers
    net1.requires_grad_(True)
    net2.requires_grad_(True)
    net1.layer1.requires_grad_(False)
    net2.layer1.requires_grad_(False)
    net1.layer2.requires_grad_(False)
    net2.layer2.requires_grad_(False)
    print('models loaded')
    
    optimizer = torch.optim.SGD([{'params': net1.parameters()},{'params': net2.parameters()}], lr=lr, momentum=momentum, weight_decay=weight_decay)
    train_data_a, train_labels_a, test_data_a, test_labels_a = utils.load_cifar100_2_classes(class1a, class2a, shuffle=True, gray=True, seed=seed, random_split=random_split, split_order=split_order, fine=fine)
    train_data_b, train_labels_b, test_data_b, test_labels_b = utils.load_cifar100_2_classes(class1b, class2b, shuffle=True, gray=True, seed=seed, random_split=random_split, split_order=split_order, fine=fine)
    assert train_data_a.shape[0] == train_data_b.shape[0]
    cross_entropy_loss = nn.CrossEntropyLoss()
    coupling_loss = nn.MSELoss()
    batches = int(train_data_a.shape[0]/batch_size)
    print('training starting')
    print(f'Coupling Weight : {coupling_weight}')
    seeds = np.arange(epochs)

    for epoch in range(epochs):
        np.random.seed(seeds[epoch])
        shuffle = np.random.permutation(train_data_a.shape[0])
        train_data_a = train_data_a[shuffle]
        train_labels_a = train_labels_a[shuffle]
        train_data_b = train_data_b[shuffle]
        train_labels_b = train_labels_b[shuffle]
        for batch in range(batches-1):
            images_a = torch.from_numpy(train_data_a[batch*batch_size:(batch+1)*batch_size]).to(device)
            labels_a = torch.from_numpy(train_labels_a[batch*batch_size:(batch+1)*batch_size]).to(device)
            outputs_a = net1(images_a)
            images_b = torch.from_numpy(train_data_b[batch*batch_size:(batch+1)*batch_size]).to(device)
            labels_b = torch.from_numpy(train_labels_b[batch*batch_size:(batch+1)*batch_size]).to(device)
            outputs_b = net2(images_b)
            loss = cross_entropy_loss(outputs_a, labels_a) + cross_entropy_loss(outputs_b, labels_b) + coupling_weight*coupling_loss(net1.fc2.weight, net2.fc2.weight) + coupling_weight*coupling_loss(net1.fc2.bias, net2.fc2.bias) + coupling_weight*coupling_loss(net1.fc1.weight, net2.fc1.weight) + coupling_weight*coupling_loss(net1.fc1.bias, net2.fc1.bias) + coupling_weight*coupling_loss(net1.fc.weight, net2.fc.weight) + coupling_weight*coupling_loss(net1.fc.bias, net2.fc.bias)
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
        loss = cross_entropy_loss(outputs_a, labels_a) + cross_entropy_loss(outputs_b, labels_b) + coupling_weight*coupling_loss(net1.fc2.weight, net2.fc2.weight) + coupling_weight*coupling_loss(net1.fc2.bias, net2.fc2.bias) + coupling_weight*coupling_loss(net1.fc1.weight, net2.fc1.weight) + coupling_weight*coupling_loss(net1.fc1.bias, net2.fc1.bias) + coupling_weight*coupling_loss(net1.fc.weight, net2.fc.weight) + coupling_weight*coupling_loss(net1.fc.bias, net2.fc.bias)
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
            if epoch%10 == 0:
                print('(Test Set) Epoch: [{}/{}], Train Loss: {}, Accuracy Task A: {}, Accuracy Task B: {}'.format(epoch+1, epochs, loss.item(), test_accuracy_a, test_accuracy_b))
            del images_a, labels_a, outputs_a, images_b, labels_b, outputs_b
            torch.cuda.empty_cache()
        net1.train()
        net2.train()
        # Save Models
        if best_accuracy < (test_accuracy_a+test_accuracy_b)/2:
            print(f'Best Accuracy on Task A: {test_accuracy_a}, Best Accuracy on Task B: {test_accuracy_b}')
            best_accuracy = (test_accuracy_a+test_accuracy_b)/2
            best_epoch = epoch+1
            best_accuracy_a = test_accuracy_a
            best_accuracy_b = test_accuracy_b
            torch.save({'epoch': epoch, 'model_state_dict': net1.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy_a}, save_path +'lenet_coupled_'+str(class1a)+'vs'+str(class2a)+ '_coupling_' + str(coupling_weight)+'.pth')
            torch.save({'epoch': epoch, 'model_state_dict': net2.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy_b}, save_path +'lenet_coupled_'+str(class1b)+'vs'+str(class2b)+ '_coupling_' + str(coupling_weight)+'.pth')

    return best_accuracy_a, best_accuracy_b, best_epoch

def resnet_coupling(checkpoint, class1a, class1b, class2a, class2b, save_path = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/coupled/', epochs=200, batch_size = 128, lr=0.01, momentum=0, weight_decay=0, coupling_weight = 1, random_split = False, seed=42, split_order=None, fine=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.get_device_name())
    torch.cuda.empty_cache()
    torch.manual_seed(seed)

    net1 = models.resnet18vw(32)
    net1.fc = nn.Linear(256, 2)
    net2 = models.resnet18vw(32)
    net2.fc = nn.Linear(256, 2)
    net1.load_state_dict(torch.load(checkpoint)['model_state_dict'])
    net2.load_state_dict(torch.load(checkpoint)['model_state_dict'])
    print(f'Accuracy A : {torch.load(checkpoint)["test_accuracy_a"]}, Accuracy B : {torch.load(checkpoint)["test_accuracy_b"]}')
    net1.to(device)
    net2.to(device)

    best_accuracy = 0
    best_accuracy_a = 0
    best_accuracy_b = 0
    best_epoch = 0

    net1.requires_grad_(False)
    net2.requires_grad_(False)
    net1.fc.requires_grad_(True)
    net2.fc.requires_grad_(True)
    net1.layer4.requires_grad_(True)
    net2.layer4.requires_grad_(True)
 
    print('models loaded')
    print(f'coupling weight: {coupling_weight}')
    
    optimizer = torch.optim.SGD([{'params': net1.parameters()},{'params': net2.parameters()}], lr=lr, momentum=momentum, weight_decay=weight_decay)
    train_data_a, train_labels_a, test_data_a, test_labels_a = utils.load_cifar100_2_classes(class1a, class2a, gray=False, seed=seed, fine=fine, random_split=random_split, split_order=split_order)
    train_data_b, train_labels_b, test_data_b, test_labels_b = utils.load_cifar100_2_classes(class1b, class2b, gray=False, seed=seed, fine=fine, random_split=random_split, split_order=split_order)
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
            loss = cross_entropy_loss(outputs_a, labels_a) + cross_entropy_loss(outputs_b, labels_b) + coupling_weight*(coupling_loss(net1.fc.weight, net2.fc.weight) + coupling_loss(net1.layer4[0].conv1.weight, net2.layer4[0].conv1.weight) + coupling_loss(net1.layer4[0].conv2.weight, net2.layer4[0].conv2.weight) + coupling_loss(net1.layer4[1].conv1.weight, net2.layer4[1].conv1.weight) + coupling_loss(net1.layer4[1].conv2.weight, net2.layer4[1].conv2.weight) + coupling_loss(net1.layer4[0].downsample[0].weight, net2.layer4[0].downsample[0].weight))
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
        loss = cross_entropy_loss(outputs_a, labels_a) + cross_entropy_loss(outputs_b, labels_b) + coupling_weight*(coupling_loss(net1.fc.weight, net2.fc.weight) + coupling_loss(net1.layer4[0].conv1.weight, net2.layer4[0].conv1.weight) + coupling_loss(net1.layer4[0].conv2.weight, net2.layer4[0].conv2.weight) + coupling_loss(net1.layer4[1].conv1.weight, net2.layer4[1].conv1.weight) + coupling_loss(net1.layer4[1].conv2.weight, net2.layer4[1].conv2.weight) + coupling_loss(net1.layer4[0].downsample[0].weight, net2.layer4[0].downsample[0].weight))
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
            if (epoch)%10 == 0:
                print('(Test Set) Epoch: [{}/{}], Train Loss: {}, Accuracy Task A: {}, Accuracy Task B: {}'.format(epoch+1, epochs, loss.item(), test_accuracy_a, test_accuracy_b))
            del images_a, labels_a, outputs_a, images_b, labels_b, outputs_b
            torch.cuda.empty_cache()
        net1.train()
        net2.train()
        # Save Models
        if best_accuracy < (test_accuracy_a+test_accuracy_b)/2:
            print(f'Best Accuracy on Task A: {test_accuracy_a}, Best Accuracy on Task B: {test_accuracy_b}')
            best_accuracy = (test_accuracy_a+test_accuracy_b)/2
            best_epoch = epoch+1
            best_accuracy_a = test_accuracy_a
            best_accuracy_b = test_accuracy_b
            torch.save({'epoch': epoch, 'model_state_dict': net1.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy_a}, save_path +'resnet_coupled_'+str(class1a)+'vs'+str(class2a)+ '_coupling_' + str(coupling_weight)+'.pth')
            torch.save({'epoch': epoch, 'model_state_dict': net2.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'test_accuracy': test_accuracy_b}, save_path +'resnet_coupled_'+str(class1b)+'vs'+str(class2b)+ '_coupling_' + str(coupling_weight)+'.pth')

    return best_accuracy_a, best_accuracy_b, best_epoch


if __name__ == '__main__':
    # Task a: rose (71) vs lion (44)
    # Task b: tulip (93) vs lepoard (43)
    coupling_weights = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
    seeds = np.arange(5)
    results = []
    print('LeNet Coupling')
    for seed in seeds:
        print(f'Seed: {seed}')
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        split_order = np.random.permutation(600)
        print('Individual Training A')
        individual_accuracy_a = di_utils.cifar100_individual_train('lenet', class1=71, class2=44, epochs=1000, batch_size=128, lr=0.01, save_path='/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/fine/', seed=seed, momentum=0, weight_decay=0, random_split=True, split_order=split_order, fine=True)
        print('Individual Training B')
        individual_accuracy_b = di_utils.cifar100_individual_train('lenet', class1=93, class2=43, epochs=1000, batch_size=128, lr=0.01, save_path='/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/fine/', seed=seed, momentum=0, weight_decay=0, random_split=True, split_order=split_order, fine=True)
        print('Joint Training')
        joint_accuracy_a, joint_accuracy_b = di_utils.cifar100_joint_train('lenet', class1a=71, class2a=44, class1b=93, class2b=43, epochs=1000, batch_size=128, lr=0.01, save_path='/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/fine/', seed=seed, momentum=0, weight_decay=0, random_split=True, split_order=split_order, fine=True)
        results.append([seed, individual_accuracy_a, individual_accuracy_b, joint_accuracy_a, joint_accuracy_b])
        np.savetxt('/nfs/ghome/live/ajain/checkpoints/di_cifar100/coupled/fine/lenet_coupling.csv', results, delimiter=', ', fmt='% s')
        print('Coupling')
        for coupling_weight in coupling_weights:
            coupling_a, coupling_b, epoch = lenet_coupling('/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/fine/lenet_joint_71_93vs44_43.pth', class1a=71, class2a=44, class1b=93, class2b=43, coupling_weight=coupling_weight, epochs=3000, batch_size=128, lr=1e-4, save_path='/nfs/ghome/live/ajain/checkpoints/di_cifar100/coupled/fine/', seed=seed, momentum=0, weight_decay=0, random_split=True, split_order=split_order, fine=True)
            results.append([seed, coupling_weight, coupling_a, coupling_b, epoch])
            np.savetxt('/nfs/ghome/live/ajain/checkpoints/di_cifar100/coupled/fine/lenet_coupling.csv', results, delimiter=', ', fmt='% s')
