import torch
import numpy as np
import time
import di_utils
import os

def lenet_cifar100_hyperparameter_search(seed):
    print('seed: {}'.format(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    epoch_range = np.array([250, 500, 1000, 2000, 5000, 10000])
    lr_range = np.array([0.0003, 0.001, 0.003, 0.01, 0.03])
    reductions = np.array([0.1, 1])
    results = [] #2*5*5*3
    class1a = 71
    class1b = 93
    class2a = 44
    class2b = 43

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    split_order = np.random.permutation(600)
    for reduction in reductions:
        for epoch in epoch_range:
            for lr in lr_range:
                print('reduction: {}, epoch: {}, lr: {}'.format(reduction, epoch, lr))
                start_time = time.time()
                accuracy_a = di_utils.cifar100_individual_train('lenet', class1=class1a, class2=class2a, seed=seed, save_path = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/hyperparameter_tuning/', epochs=epoch, batch_size=int(128*reduction), lr=lr, momentum=0, weight_decay=0, random_split=True, split_order=split_order, fine=True, reduction=reduction)
                accuracy_b = di_utils.cifar100_individual_train('lenet', class1=class1b, class2=class2b, seed=seed, save_path = '/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/hyperparameter_tuning/', epochs=epoch, batch_size=int(128*reduction), lr=lr, momentum=0, weight_decay=0, random_split=True, split_order=split_order, fine=True, reduction=reduction)
                end_time = time.time()
                results.append([reduction, epoch, lr, accuracy_a, accuracy_b, end_time-start_time])
                np.savetxt(f'/nfs/ghome/live/ajain/checkpoints/di_cifar100/baseline/hyperparameter_tuning/lenet_seed{seed}.csv', results, delimiter=', ', fmt='% s')
                print(results[-1])

if __name__ == '__main__':
    seed = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    lenet_cifar100_hyperparameter_search(seed)