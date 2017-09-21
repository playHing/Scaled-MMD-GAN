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

class MMDCE_GAN(MMD_GAN):
        
    def set_loss(self, G, images):
        with tf.variable_scope("discriminator") as scope:
            G1 = linear(G, 1, 'd_htop_lin')
            scope.reuse_variables()
            images1 = linear(images, 1, 'd_htop_lin')
        # no need to ouput sigmoids, loss function below takes logits
        self.gan_ce_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=G1, labels=tf.zeros_like(G1))) # fake
        self.gan_ce_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=images1, labels=tf.ones_like(images1))) #real

        super(MMDCE_GAN, self).set_loss(G, images)
        self.optim_name = 'kernel+cross_entropy_loss'
        

    def add_gradient_penalty(self, kernel, fake_data, real_data):
        alpha = tf.random_uniform(shape=[self.batch_size, 1])
        if 'mid' in self.config.suffix:
            alpha = .4 + .2 * alpha
        elif 'edges' in self.config.suffix:
            qq = tf.cast(tf.reshape(tf.multinomial([[.5, .5]], self.batch_size),
                                    [self.batch_size, 1]), tf.float32)
            alpha = .1 * alpha * qq + (1. - .1 * alpha) * (1. - qq)
        elif 'edge' in self.config.suffix:
            alpha = .99 + .01 * alpha
        x_hat = (1. - alpha) * real_data + alpha * fake_data
        Ekx = lambda yy: tf.reduce_mean(kernel(x_hat, yy, K_XY_only=True), axis=1)
        witness = Ekx(real_data) - Ekx(fake_data)
        gradients = tf.gradients(witness, [x_hat])[0]
        penalty = tf.reduce_mean(tf.square(tf.norm(gradients, axis=1) - 1.0))
        
        print('adding gradient penalty')
        # We need to:
        #  - minimize MMD wrt generator
        #  - maximize MMD wrt discriminator
        #  - minimize GAN cross-entropy wrt discriminator
        if self.config.gradient_penalty > 0:
            self.gp = tf.get_variable('gradient_penalty', dtype=tf.float32,
                                      initializer=self.config.gradient_penalty)
            self.g_loss = self.mmd_loss
            self.d_loss = -self.mmd_loss + penalty * self.gp + self.gan_ce_loss
            self.optim_name += ' gp %.1f' % self.config.gradient_penalty
        else:
            self.g_loss = self.mmd_loss
            self.d_loss = -self.mmd_loss + self.gan_ce_loss
        variable_summaries([(gradients, 'dx_gradients')])
        tf.summary.scalar(self.optim_name + ' G', self.g_loss)
        tf.summary.scalar(self.optim_name + ' D', self.d_loss)
        tf.summary.scalar('dx_penalty', penalty)
