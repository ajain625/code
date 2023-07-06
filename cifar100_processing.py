import numpy as np
import pickle

f = open('/nfs/ghome/live/ajain/datasets/cifar100/cifar-100-python/test', 'rb')
dict = pickle.load(f, encoding='bytes')
print(dict.keys())
images = dict[b'data']
labels = dict[b'coarse_labels']
imagearray = np.array(images, dtype=np.float32)
labelarray = np.array(labels)
for label in range(20):
    class_data = imagearray[labelarray == label]/255
    class_data = np.reshape(class_data, (class_data.shape[0], 3, 32, 32))
    np.save('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/'+'test'+str(label+1), class_data)
