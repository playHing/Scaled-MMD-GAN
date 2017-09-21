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
        
    def set_loss(self, G, images):
        assert self.config.architecture == 'dcgan1', 'Discriminator must return a single feature.'
        # with tf.variable_scope("discriminator") as scope:
        #     G1 = linear(G, 1, 'd_htop_lin')
        #     scope.reuse_variables()
        #     images1 = linear(images, 1, 'd_htop_lin')
        # no need to ouput sigmoids, loss function below takes logits
        gan_ce_fake =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=G, labels=tf.zeros_like(G1))) # fake
        gan_ce_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=images, labels=tf.ones_like(images1))) #real
                
        self.d_loss = (gan_ce_fake + gan_ce_real)/2.0
        self.g_loss = -gan_ce_fake
        self.optim_name = 'gan_loss'
        
        tf.summary.scalar(self.optim_name + ' G', self.g_loss)
        tf.summary.scalar(self.optim_name + ' D', self.d_loss)
