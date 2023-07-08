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
        #imagearray = (imagearray- imagearray.mean(axis=(2, 3)))/imagearray.std(axis=(2, 3))
        imagearray = imagearray/255.0
    if gray:
        imagearray = imagearray.mean(axis=1, keepdims=True)
    
    return imagearray, labelarray

def cifar_accuracy(gt, output_probs):
    predictions = np.argmax(output_probs, axis=1)
    #print(predictions[:2])
    return np.sum(gt == predictions)/len(gt)

def normalise_images(images, gray=False, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    images = np.transpose(images, (2,3,0,1))
    images = (images - np.mean(images, axis=(0,1), keepdims=True))/np.std(images, axis=(0,1), keepdims=True)
    images = images*std + mean
    images = np.transpose(images, (2,3,0,1))
    if gray:
        return images.mean(axis=1, keepdims=True).astype(np.float32)
    return images.astype(np.float32)

def load_cifar100_2_classes(class1, class2, gray=False, shuffle=True, seed=42):
    train_data = normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class1)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/train'+str(class2)+'.npy')]), gray=gray)
    train_labels = np.hstack([np.zeros(len(train_data)//2, dtype=int), np.ones(len(train_data)//2, dtype=int)])
    test_data = normalise_images(np.vstack([np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class1)+'.npy'), np.load('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/test'+str(class2)+'.npy')]), gray=gray)
    test_labels = np.hstack([np.zeros(len(test_data)//2, dtype=int), np.ones(len(test_data)//2, dtype=int)])
    if shuffle:
        np.random.seed(seed)
        train_shuffle = np.random.permutation(len(train_data))
        train_data = train_data[train_shuffle]
        train_labels = train_labels[train_shuffle]
    return train_data, train_labels, test_data, test_labels

train_data, train_labels, test_data, test_labels = load_cifar100_2_classes(1, 2)
plt.imshow(np.transpose(train_data[1], (1, 2, 0)))
plt.savefig('test.png')
#print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
#print(train_data.mean(axis=(0,2,3)), train_data.std(axis=(0,2,3)))
#print(test_data.mean(axis=(0,2,3)), test_data.std(axis=(0,2,3)))