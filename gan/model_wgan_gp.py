from __future__ import division, print_function
from glob import glob
import os
import time

import numpy as np
import scipy.misc
from six.moves import xrange
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import lmdb
import io
import sys
from IPython.display import display


from model_mmd2 import MMD_GAN, tf, np
import mmd as MMD
import load
from ops import batch_norm, conv2d, deconv2d, linear, lrelu
from utils import save_images, unpickle, read_and_scale, center_and_scale, variable_summaries, conv_sizes, pp
import pprint

class GAN(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        config.dof_dim = 1
        super(GAN, self).__init__(sess, config, **kwargs)
        
    def set_loss(self, G, images):
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1])
        real_data = self.images
        fake_data = self.G
        differences = fake_data - real_data
        interpolates0 = real_data + (alpha*differences)
        interpolates = self.discriminator(interpolates0, reuse=True)

        # with tf.variable_scope("discriminator") as scope:
        #     G1 = linear(G, 1, 'd_htop_lin')
        #     scope.reuse_variables()
        #     images1 = linear(images, 1, 'd_htop_lin')
        #     interpolates1 = linear(interpolates, 1, 'd_htop_lin')

        gradients = tf.gradients(interpolates, [interpolates0])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        self.gp = tf.get_variable('gradient_penalty', dtype=tf.float32,
                                  initializer=self.config.gradient_penalty)

        self.d_loss = tf.reduce_mean(G) - tf.reduce_mean(images) + self.gp * gradient_penalty
        self.g_loss = -tf.reduce_mean(G)
        self.optim_name = 'wgan_gp_loss'

        tf.summary.scalar(self.optim_name + ' G', self.g_loss)
        tf.summary.scalar(self.optim_name + ' D', self.d_loss)
