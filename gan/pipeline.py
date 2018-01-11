#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:11:46 2018

@author: mikolajbinkowski
"""
import os, time, lmdb, io
import numpy as np
import tensorflow as tf
import utils
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

class Pipeline:
    def __init__(self, output_size, c_dim, batch_size, data_dir, **kwargs):
        self.output_size = output_size
        self.c_dim = c_dim

#        data_dir = os.path.join(self.data_dir, self.dataset)
        self.batch_size = batch_size
        self.read_batch = max(4000, batch_size * 10)
        self.read_count = 0
        self.data_dir = data_dir
        self.shape = [self.read_batch, self.output_size, self.output_size, self.c_dim]
    
    def _transform(self, x):
        return x
    
    def connect(self):
        assert hasattr(self, 'single_sample'), 'Pipeline needs to have single_sample defined before connecting'
        self.single_sample.set_shape(self.shape)
        ims = tf.train.shuffle_batch([self.single_sample], self.batch_size,
                                    capacity=self.read_batch,
                                    min_after_dequeue=self.read_batch//8,
                                    num_threads=16,
                                    enqueue_many=len(self.shape) == 4)
        return self._transform(ims)
    

class LMDB(Pipeline):
    def __init__(self, *args, timer=None, **kwargs):
        print(*args)
        print(**kwargs)
        super(LMDB, self).__init__(*args, **kwargs)
        self.timer = timer
        self.keys = []
        env = lmdb.open(self.data_dir, map_size=1099511627776, max_readers=100, readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            while cursor.next():
                self.keys.append(cursor.key())
        print('Number of records in lmdb: %d' % len(self.keys))
        env.close()
        # tf queue for getting keys
        key_producer = tf.train.string_input_producer(self.keys, shuffle=True)
        single_key = key_producer.dequeue()
        self.single_sample = tf.py_func(self._get_sample_from_lmdb, [single_key], tf.float32)
        
        
    def _get_sample_from_lmdb(self, key, limit=None):
        if limit is None:
            limit = self.read_batch
        with tf.device('/cpu:0'):
            rc = self.read_count
            self.read_count += 1
            tt = time.time()
            self.timer(rc, 'read start')
            env = lmdb.open(self.data_dir, map_size=1099511627776, max_readers=100, readonly=True)
            ims = []
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                cursor.set_key(key)
                while len(ims) < limit:
                    key, byte_arr = cursor.item()
                    byte_im = io.BytesIO(byte_arr)
                    byte_im.seek(0)
                    try:
                        im = Image.open(byte_im)
                        ims.append(utils.center_and_scale(im, size=self.output_size))
                    except Exception as e:
                        print(e)
                    if not cursor.next():
                        cursor.first()
            env.close()
            self.timer(rc, 'read time = %f' % (time.time() - tt))
            return np.asarray(ims, dtype=np.float32)       
     
        
    def constant_sample(self, size):
        choice = np.random.choice(self.keys, 1)[0]
        return self._get_sample_from_lmdb(choice, limit=size)


class JPEG(Pipeline):
    def __init__(self, *args, base_size=160, random_crop=9, **kwargs):
        super(JPEG, self).__init__(*args, **kwargs)
        files = glob(os.path.join(self.data_dir, '*.jpg'))

        filename_queue = tf.train.string_input_producer(files, shuffle=True)
        reader = tf.WholeFileReader()
        _, raw = reader.read(filename_queue)
        decoded = tf.image.decode_jpeg(raw, channels=self.c_dim) # HWC
        bs = base_size + 2 * random_crop
        cropped = tf.image.resize_image_with_crop_or_pad(decoded, bs, bs)
        if random_crop > 0:
            cropped = tf.image.random_flip_left_right(cropped)
            cropped = tf.random_crop(cropped, [base_size, base_size, self.c_dim])
        self.single_sample = cropped
        self.shape = [base_size, base_size, self.c_dim]    
        
    def _transform(self, x):
        x = tf.image.resize_bilinear(x, (self.output_size, self.output_size))
        return tf.cast(x, tf.float32)/255.


class Mnist(Pipeline):
    def __init__(self, *args, **kwargs):
        super(Mnist, self).__init__(*args, **kwargs)
        fd = open(os.path.join(self.data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)
    
        fd = open(os.path.join(self.data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)
    
        fd = open(os.path.join(self.data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)
    
        fd = open(os.path.join(self.data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)
    
        trY = np.asarray(trY)
        teY = np.asarray(teY)
    
        X = np.concatenate((trX, teX), axis=0).astype(np.float32) / 255.
        y = np.concatenate((trY, teY), axis=0)
    
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
    
        queue = tf.train.input_producer(tf.constant(X), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)


class Cifar10(Pipeline):
    def __init__(self, *args, **kwargs):
        super(Cifar10, self).__init__(*args, **kwargs)
        categories = np.arange(10)
        batchesX, batchesY = [], []
        for batch in range(1,6):
            loaded = utils.unpickle(os.path.join(self.data_dir, 'data_batch_%d' % batch))
            idx = np.in1d(np.array(loaded['labels']), categories)
            batchesX.append(loaded['data'][idx].reshape(idx.sum(), 3, 32, 32))
            batchesY.append(np.array(loaded['labels'])[idx])
        trX = np.concatenate(batchesX, axis=0).transpose(0, 2, 3, 1)
        trY = np.concatenate(batchesY, axis=0)
        
        test = utils.unpickle(os.path.join(self.data_dir, 'test_batch'))
        idx = np.in1d(np.array(test['labels']), categories)
        teX = test['data'][idx].reshape(idx.sum(), 3, 32, 32).transpose(0, 2, 3, 1)
        teY = np.array(test['labels'])[idx]
    
        X = np.concatenate((trX, teX), axis=0).astype(np.float32) / 255.
        y = np.concatenate((trY, teY), axis=0)
    
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)

        queue = tf.train.input_producer(tf.constant(X), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)


class SingleFile(Pipeline):
    def __init__(self, *args, **kwargs):
        super(JPEG, self).__init__(*args, **kwargs)
        if self.dataset in ['mnist', 'cifar10']:
            path = os.path.join(self.data_dir, self.dataset)
            X, y = getattr(load, self.dataset)(path)
        elif self.dataset == 'GaussianMix':
            G_config = {'g_line': None}
            path = os.path.join(self.sample_dir, self.description)
            X, G_config['ax1'], G_config['writer'] = load.GaussianMix(path)
            G_config['fig'] = G_config['ax1'].figure
            self.G_config = G_config
        else:
            raise ValueError("not implemented dataset '%s'" % self.dataset)

        queue = tf.train.input_producer(tf.constant(X.astype(np.float32)), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)


class GaussianMix(Pipeline):
    def __init__(self, *args, sample_dir='/', means=[.0, 3.0], stds=[1.0, .5], size=1000, **kwargs):
        super(GaussianMix, self).__init__(*args, **kwargs)
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
        self.G_config = {'g_line': None,
                        'ax1': ax1,
                        'writer': wrtr,
                        'figure': ax1.figure}
        queue = tf.train.input_producer(tf.constant(X_real.astype(np.float32)), shuffle=False)
        self.single_sample = queue.dequeue_many(self.read_batch)

        
def myhist(X, ax=plt, bins='auto', **kwargs):
    hist, bin_edges = np.histogram(X, bins=bins)
    hist = hist / hist.max()
    return ax.plot(
        np.c_[bin_edges, bin_edges].ravel(),
        np.r_[0, np.c_[hist, hist].ravel(), 0],
        **kwargs
    )


            
#            
#    def set_input_pipeline(self, streams=None):
#        if self.dataset in ['mnist', 'cifar10']:
#            path = os.path.join(self.data_dir, self.dataset)
#            data_X, data_y = getattr(load, self.dataset)(path)
#        elif self.dataset in ['celebA']:
#            files = glob(os.path.join(self.data_dir, self.dataset, '*.jpg'))
#            data_X = np.array([utils.get_image(f, 160, 160, resize_height=self.output_size, 
#                                               resize_width=self.output_size) for f in files[:]])
#        elif self.dataset == 'GaussianMix':
#            G_config = {'g_line': None}
#            path = os.path.join(self.sample_dir, self.description)
#            data_X, G_config['ax1'], G_config['writer'] = load.GaussianMix(path)
#            G_config['fig'] = G_config['ax1'].figure
#            self.G_config = G_config
#        else:
#            raise ValueError("not implemented dataset '%s'" % self.dataset)
#        if streams is None:
#            streams = [self.real_batch_size]
#        streams = np.cumsum(streams)
#        bs = streams[-1]
#
#        queue = tf.train.input_producer(tf.constant(data_X.astype(np.float32)), 
#                                                shuffle=False)
#        single_sample = queue.dequeue_many(bs * 4)
#        single_sample.set_shape([bs * 4, self.output_size, self.output_size, self.c_dim])
#        ims = tf.train.shuffle_batch(
#            [single_sample], 
#            batch_size=bs,
#            capacity=max(bs * 8, self.batch_size * 32),
#            min_after_dequeue=max(bs * 2, self.batch_size * 8),
#            num_threads=4,
#            enqueue_many=True
#        )
#        
#        self.images = ims[:streams[0]]
#
#        for j in np.arange(1, len(streams)):
#            self.__dict__.update({'images%d' % (j + 1): ims[streams[j - 1]: streams[j]]})
#        off = int(np.random.rand()*( data_X.shape[0] - self.batch_size*2))
#        self.additional_sample_images = data_X[off: off + self.batch_size].astype(np.float32)
#
#        
#    def set_input3_pipeline(self, streams=None):
#        if streams is None:
#            streams = [self.real_batch_size]
#        streams = np.cumsum(streams)
#        bs = streams[-1]
#        read_batch = max(20000, bs * 10)
#
#        self.files = glob(os.path.join(self.data_dir, self.dataset, '*.jpg'))
#        self.read_count = 0
#        def get_read_batch(k, limit=read_batch):
#            with tf.device('/cpu:0'):
#                rc = self.read_count
#                self.read_count += read_batch
#                if rc//len(self.files) < self.read_count//len(self.files):
#                    self.files = list(np.random.permutation(self.files))
#                tt = time.time()
#                self.timer(rc, 'read start')
#                ims = []
#                files_k = self.files[k: k + read_batch] + self.files[: max(0, k + read_batch - len(self.files))]
#                for ii, ff in enumerate(files_k):
#                    ims.append(utils.get_image(ff, 160, 160, resize_height=self.output_size, 
#                                               resize_width=self.output_size))
#                self.timer(rc, 'read time = %f' % (time.time() - tt))
#                return np.asarray(ims, dtype=np.float32)                
#                
#
#        choice = np.random.choice(len(self.files), 1)[0]
#        sampled = get_read_batch(choice, self.sample_size + self.batch_size)
#
#        self.additional_sample_images = sampled[self.sample_size: self.sample_size + self.batch_size]
#        print('self.additional_sample_images.shape: ' + repr(self.additional_sample_images.shape))
#        # tf queue for getting keys
##        key_producer = tf.train.string_input_producer(keys, shuffle=True)
#        key_producer = tf.train.range_input_producer(len(self.files), shuffle=True)
#        single_key = key_producer.dequeue()
#        
#        single_sample = tf.py_func(get_read_batch, [single_key], tf.float32)
#        single_sample.set_shape([read_batch, self.output_size, self.output_size, self.c_dim])
#
##        self.images = tf.train.shuffle_batch([single_sample], self.batch_size, 
##                                            capacity=read_batch * 4, 
##                                            min_after_dequeue=read_batch//2,
##                                            num_threads=2,
##                                            enqueue_many=True)
#        ims = tf.train.shuffle_batch([single_sample], bs,
#                                            capacity=read_batch * 16,
#                                            min_after_dequeue=read_batch * 2,
#                                            num_threads=8,
#                                            enqueue_many=True)
#        
#        self.images = ims[:streams[0]]
#        for j in np.arange(1, len(streams)):
#            self.__dict__.update({'images%d' % (j + 1): ims[streams[j - 1]: streams[j]]})
# 
#    def set_jpeg_pipeline(self, streams=None):
#        if streams is None:
#            streams = [self.real_batch_size]
#        streams = np.cumsum(streams)
#        files = glob(os.path.join(self.data_dir, self.dataset, '*.jpg'))
#        ims = utils.tf_read_jpeg(files, 
#                               base_size=160, target_size=self.output_size, 
#                               batch_size=streams[-1], 
#                               capacity=4000, num_threads=4)
#        self.images = ims[:streams[0]]
#        for j in np.arange(1, len(streams)):
#            self.__dict__.update({'images%d' % (j + 1): ims[streams[j - 1]: streams[j]]})
#
#            
#    def set_tf_records_pipeline(self, streams=None):     
#        if streams is None:
#            streams = [self.real_batch_size]
#        streams = np.cumsum(streams)
#        bs = streams[-1]
#        
#        path = '/nfs/data/dougals/'
#        if not os.path.exists(path):
#            path = self.config.data_dir
#            
#        with tf.device(tf.train.replica_device_setter(0, worker_device='/cpu:0')):
#            filename_queue = tf.train.string_input_producer(
#                tf.gfile.Glob(os.path.join(path, 'lsun-32/bedroom_train_*')), num_epochs=None)
#            reader = tf.TFRecordReader()
#            _, serialized_example = reader.read(filename_queue)
#            features = tf.parse_single_example(serialized_example, features={
#                'image/class/label': tf.FixedLenFeature([1], tf.int64),
#                'image/encoded': tf.FixedLenFeature([], tf.string),
#            })
#            image = tf.image.decode_jpeg(features['image/encoded'])
#            single_sample = tf.cast(image, tf.float32)/255.
#            single_sample.set_shape([self.output_size, self.output_size, self.c_dim])
#
#            ims = tf.train.shuffle_batch([single_sample], bs,
#                                         capacity=bs * 8,
#                                         min_after_dequeue=bs * 2, 
#                                         num_threads=16,
#                                         enqueue_many=False)
#        self.images = ims[:streams[0]]
#        for j in np.arange(1, len(streams)):
#            self.__dict__.update({'images%d' % (j + 1): ims[streams[j - 1]: streams[j]]})
#            
