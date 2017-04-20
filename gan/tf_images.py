#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:19:11 2017

@author: mikolajbinkowski
"""

import os
from PIL import Image
import numpy as np
#import tensorflow as tf
size = 64.
files = os.listdir(os.path.join(os.getcwd(), 'data/lsun_small/'))
files = [os.path.join('./data/lsun_small/', f) for f in files]
file = files[0]

def xx(file):
    im = Image.open(file)
    arr = np.array(im)
    scale = min(im.size)/size
    new_size = np.array(im.size)/scale
    im.thumbnail(new_size)
#    im.show()
    arr = np.array(im)
    l0 = (arr.shape[0] - size)//2
    l1 = (arr.shape[1] - size)//2
    return arr[l0:l0 + size, l1: l1 + size, :]

Arr = np.concatenate([[xx(f)] for f in files], axis =0)



#filename_queue=tf.train.string_input_producer(files[:1], capacity=128)
#
#reader=tf.WholeFileReader()
#filename, content = reader.read(filename_queue)
#images=tf.image.decode_jpeg(content, channels=3)
#images=tf.cast(images, tf.float32)
#images = tf.image.resize_image_with_crop_or_pad(images, 256, 256)
#resized_images=tf.image.resize_images(images, [size, size])
from PIL import Image
import lmdb
import io

env = lmdb.open('./PhD/MMD/opt-mmd/gan/data/bedroom_val_lmdb', map_size=1099511627776,
                    max_readers=100, readonly=True)
txn = env.begin(write=False)
cursor = txn.cursor()
i = 0
for k, byte_arr in cursor:
    if i > 0:
        break
    im = Image.open(io.BytesIO(byte_arr))
    i += 1

tempBuff = StringIO()
tempBuff.write(v)
tempBuff.seek(0)

def export_images(db_path, out_dir, flat=False, limit=-1):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            if not flat:
                image_out_dir = join(out_dir, '/'.join(key[:6]))
            else:
                image_out_dir = out_dir
            if not exists(image_out_dir):
                os.makedirs(image_out_dir)
            image_out_path = join(image_out_dir, key + '.webp')
            with open(image_out_path, 'w') as fp:
                fp.write(val)
            count += 1
            if count == limit:
                break
            if count % 1000 == 0:
                print('Finished', count, 'images')
                