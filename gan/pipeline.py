##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Thu Jan 11 14:11:46 2018
#
#@author: mikolajbinkowski
#"""
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
#    def set_lmdb_pipeline(self, streams=None):
#        if streams is None:
#            streams = [self.real_batch_size]
#        streams = np.cumsum(streams)
#        bs = streams[-1]
#
#        data_dir = os.path.join(self.data_dir, self.dataset)
#        keys = []
#        read_batch = max(4000, self.real_batch_size * 10)
#        # getting keys in database
#        env = lmdb.open(data_dir, map_size=1099511627776, max_readers=100, readonly=True)
#        with env.begin() as txn:
#            cursor = txn.cursor()
#            while cursor.next():
#                keys.append(cursor.key())
#        print('Number of records in lmdb: %d' % len(keys))
#        env.close()
#        
#        # value [np.array] reader for given key
#        self.read_count = 0
#        def get_sample_from_lmdb(key, limit=read_batch):
#            with tf.device('/cpu:0'):
#                rc = self.read_count
#                self.read_count += 1
#                tt = time.time()
#                self.timer(rc, 'read start')
#                env = lmdb.open(data_dir, map_size=1099511627776, max_readers=100, readonly=True)
#                ims = []
#                with env.begin(write=False) as txn:
#                    cursor = txn.cursor()
#                    cursor.set_key(key)
#                    while len(ims) < limit:
#                        key, byte_arr = cursor.item()
#                        byte_im = io.BytesIO(byte_arr)
#                        byte_im.seek(0)
#                        try:
#                            im = Image.open(byte_im)
#                            ims.append(utils.center_and_scale(im, size=self.output_size))
#                        except Exception as e:
#                            print(e)
#                        if not cursor.next():
#                            cursor.first()
#                env.close()
#                self.timer(rc, 'read time = %f' % (time.time() - tt))
#                return np.asarray(ims, dtype=np.float32)
#
#        choice = np.random.choice(keys, 1)[0]
#        sampled = get_sample_from_lmdb(choice, self.sample_size + self.batch_size)
#
#        self.additional_sample_images = sampled[self.sample_size: self.sample_size + self.batch_size]
#        print('self.additional_sample_images.shape: ' + repr(self.additional_sample_images.shape))
#        # tf queue for getting keys
#        key_producer = tf.train.string_input_producer(keys, shuffle=True)
#        single_key = key_producer.dequeue()
#        
#        single_sample = tf.py_func(get_sample_from_lmdb, [single_key], tf.float32)
#        single_sample.set_shape([read_batch, self.output_size, self.output_size, self.c_dim])
#
##        self.images = tf.train.shuffle_batch([single_sample], self.batch_size, 
##                                            capacity=read_batch * 4, 
##                                            min_after_dequeue=read_batch//2,
##                                            num_threads=2,
##                                            enqueue_many=True)
#        ims = tf.train.shuffle_batch([single_sample], bs,
#                                            capacity=max(bs * 8, read_batch),
#                                            min_after_dequeue=max(bs * 2, read_batch//8),
#                                            num_threads=16,
#                                            enqueue_many=True)
#        
#        self.images = ims[:streams[0]]
#        for j in np.arange(1, len(streams)):
#            self.__dict__.update({'images%d' % (j + 1): ims[streams[j - 1]: streams[j]]})