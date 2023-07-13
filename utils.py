import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_cifar_batch(batch_number, normalize=True, gray=True):
    assert batch_number in [1, 2, 3, 4, 5, 'test']
    path = '/nfs/ghome/live/ajain/datasets/cifar10/cifar-10-batches-py/'
    file = 'data_batch_'+ str(batch_number)
    if batch_number == 'test':
        file = 'test_batch'
    f = open(path+file, 'rb')
    dict = pickle.load(f, encoding='bytes')
    images = dict[b'data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict[b'labels']
    imagearray = np.array(images, dtype=np.float32)
    labelarray = np.array(labels)

    if normalize:
        imagearray = imagearray/255.0
    if gray:
        imagearray = imagearray.mean(axis=1, keepdims=True)
    
    return imagearray, labelarray

def cifar_accuracy(gt, output_probs):
    predictions = np.argmax(output_probs, axis=1)
    return np.sum(gt == predictions)/len(gt)

def normalise_images(images, gray=False, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    images = np.transpose(images, (2,3,0,1))
    images = (images - np.mean(images, axis=(0,1), keepdims=True))/np.std(images, axis=(0,1), keepdims=True)
    images = images*std + mean
    images = np.transpose(images, (2,3,0,1))
    if gray:
        return images.mean(axis=1, keepdims=True).astype(np.float32)
    return images.astype(np.float32)

def load_cifar100_2_classes(class1, class2, gray=False, shuffle=True, seed=42, random_split=False, reduction=1):
    train_data = normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class1)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class2)+'.npy')]), gray=gray)
    train_len = len(train_data)
    train_labels = np.hstack([np.zeros(train_len//2, dtype=int), np.ones(train_len//2, dtype=int)])
    test_data = normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class1)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class2)+'.npy')]), gray=gray)
    test_len = len(test_data)
    test_labels = np.hstack([np.zeros(test_len//2, dtype=int), np.ones(test_len//2, dtype=int)])
    if random_split:
        np.random.seed(seed)
        data = np.vstack([train_data, test_data])
        labels = np.hstack([train_labels, test_labels])
        shuffle = np.random.permutation(len(data))
        data = data[shuffle]
        labels = labels[shuffle]
        train_data = data[:int(train_len*reduction)]
        train_labels = labels[:int(train_len*reduction)]
        test_data = data[-test_len:]
        test_labels = labels[-test_len:]
    elif shuffle:
        np.random.seed(seed)
        train_shuffle = np.random.permutation(train_len)
        train_data = train_data[train_shuffle]
        train_data = train_data[:int(train_len*reduction)]
        train_labels = train_labels[train_shuffle]
        train_labels = train_labels[:int(train_len*reduction)]
    return train_data, train_labels, test_data, test_labels

def load_cifar100_4_classes_(class1a, class1b, class2a, class2b, gray=False, shuffle=True, seed=42, random_split=False, reduction=1):
    # Think  about how to ensure same train-test as individual train-test
    raise NotImplementedError
    train_data_a = normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class1a)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class2a)+'.npy')]), gray=gray)
    train_len_a = len(train_data_a)
    train_labels_a = np.hstack([np.zeros(train_len_a//2, dtype=int), np.ones(train_len_a//2, dtype=int)])
    train_data_b = normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class1b)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class2b)+'.npy')]), gray=gray)
    train_len_b = len(train_data_b)
    train_labels_b = np.hstack([np.zeros(train_len_b//2, dtype=int), np.ones(train_len_b//2, dtype=int)])
    test_data_a = normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class1a)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class2a)+'.npy')]), gray=gray)
    test_data_b = normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class1b)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class2b)+'.npy')]), gray=gray)
    test_len_a = len(test_data_a)
    test_len_b = len(test_data_b)
    test_labels_a = np.hstack([np.zeros(test_len_a//2, dtype=int), np.ones(test_len_a//2, dtype=int)])
    test_labels_b = np.hstack([np.zeros(test_len_b//2, dtype=int), np.ones(test_len_b//2, dtype=int)])
    if random_split:
        #TODO
    elif shuffle:
        #TODO
    

# train_data, train_labels, test_data, test_labels = load_cifar100_2_classes(1, 2)
# plt.imshow(np.transpose(train_data[1], (1, 2, 0)))
# plt.savefig('test.png')
# print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
# print(train_data.mean(axis=(0,2,3)), train_data.std(axis=(0,2,3)))
# print(test_data.mean(axis=(0,2,3)), test_data.std(axis=(0,2,3)))