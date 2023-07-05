import pickle
import numpy as np

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