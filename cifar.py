import numpy as np
#CIFAR10

def load_data():
    file = "cifar-10/data_batch_1"
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    img=[np.mean(np.array([item[:1024],item[1024:2048],item[2048:]]), axis=0) for item in dict[b'data']]
    file = "cifar-10/test_batch"
    import pickle
    with open(file, 'rb') as fo:
        dict2 = pickle.load(fo, encoding='bytes')
    img2=[np.mean(np.array([item[:1024],item[1024:2048],item[2048:]]), axis=0) for item in dict[b'data']]
    return img,np.array(dict[b'labels']),img2,np.array(dict2[b'labels'])

load_data()

