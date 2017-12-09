#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:01:03 2017

@author: mikolajbinkowski
"""

def load_mnist(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X/255.,y


def load_cifar10(self, categories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    data_dir = os.path.join(self.data_dir, self.dataset_name)

    batchesX, batchesY = [], []
    for batch in range(1,6):
        loaded = unpickle(os.path.join(data_dir, 'data_batch_%d' % batch))
        idx = np.in1d(np.array(loaded['labels']), categories)
        batchesX.append(loaded['data'][idx].reshape(idx.sum(), 3, 32, 32))
        batchesY.append(np.array(loaded['labels'])[idx])
    trX = np.concatenate(batchesX, axis=0).transpose(0, 2, 3, 1)
    trY = np.concatenate(batchesY, axis=0)
    
    test = unpickle(os.path.join(data_dir, 'test_batch'))
    idx = np.in1d(np.array(test['labels']), categories)
    teX = test['data'][idx].reshape(idx.sum(), 3, 32, 32).transpose(0, 2, 3, 1)
    teY = np.array(test['labels'])[idx]

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X/255.,y