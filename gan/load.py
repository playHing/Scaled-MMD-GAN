from __future__ import division, print_function
from glob import glob
import os

import numpy as np
import scipy.misc
from six.moves import xrange
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import lmdb
from IPython.display import display

import mmd as MMD
from ops import batch_norm, conv2d, deconv2d, linear, lrelu
from utils import unpickle

def mnist(data_dir):
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


def cifar10(data_dir, categories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
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


def load_GaussianMix(sample_dir, means=[.0, 3.0], stds=[1.0, .5], size=1000):
    from matplotlib import animation
    X_real = np.r_[
        np.random.normal(0,  1, size=size),
        np.random.normal(3, .5, size=size),
    ]   
    X_real = X_real.reshape(X_real.shape[0], 1, 1, 1)
    
    xlo = -5
    xhi = 7
    
    ax1 = plt.gca()
    fig = ax1.figure
    ax1.grid(False)
    ax1.set_yticks([], [])
    myhist(X_real.ravel(), color='r')
    ax1.set_xlim(xlo, xhi)
    ax1.set_ylim(0, 1.05)
    ax1._autoscaleXon = ax1._autoscaleYon = False
    
    wrtr = animation.writers['ffmpeg'](fps=20)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    wrtr.setup(fig=fig, outfile=os.path.join(sample_dir, 'train.mp4'), dpi=100)
    return X_real, ax1, wrtr

        
def myhist(X, ax=plt, bins='auto', **kwargs):
    hist, bin_edges = np.histogram(X, bins=bins)
    hist = hist / hist.max()
    return ax.plot(
        np.c_[bin_edges, bin_edges].ravel(),
        np.r_[0, np.c_[hist, hist].ravel(), 0],
        **kwargs
    )