import numpy as np
import pickle
import matplotlib.pyplot as plt

f = open('/nfs/ghome/live/ajain/datasets/cifar100/cifar-100-python/train', 'rb')
dict = pickle.load(f, encoding='bytes')
print(dict.keys())
images = dict[b'data']
labels = dict[b'coarse_labels']
imagearray = np.array(images, dtype=np.float32)
labelarray = np.array(labels)
for label in range(20):
    class_data = imagearray[labelarray == label]
    class_data = np.reshape(class_data, (class_data.shape[0], 3, 32, 32))/255
    #plt.imshow(np.transpose(class_data[1], (1, 2, 0)))
    #plt.savefig(f'test'+str(label+1)+'.png')
    np.save('/nfs/ghome/live/ajain/datasets/cifar100/class_wise/'+'train'+str(label+1), class_data)
