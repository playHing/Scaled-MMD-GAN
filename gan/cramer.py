import mmd as MMD
from mmd import _eps

from model_mmd2 import MMD_GAN, tf, np
from model_me_brb import MEbrb_GAN
from utils import variable_summaries
from cholesky import me_loss
from ops import batch_norm, conv2d, deconv2d, linear, lrelu
from glob import glob
import os
import time


class Cramer_GAN(MMD_GAN):                   
    def build_model(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.get_variable('lr', dtype=tf.float32, initializer=self.config.learning_rate)
        if 'lsun' in self.config.dataset:
            self.set_lmdb_pipeline(streams=[self.real_batch_size, self.real_batch_size])
        elif 'celebA' in self.config.dataset:
            self.set_folder_pipeline(streams=[self.real_batch_size, self.real_batch_size])
        else:
            self.set_input_pipeline(streams=[self.real_batch_size, self.real_batch_size])


        self.sample_z = tf.constant(np.random.uniform(-1, 1, size=(self.sample_size,
                                                      self.z_dim)).astype(np.float32),
                                    dtype=tf.float32, name='sample_z')

        self.G = self.generator(tf.random_uniform([self.batch_size, self.z_dim], minval=-1.,
                                                   maxval=1., dtype=tf.float32, name='z'))
        self.G2 = self.generator(tf.random_uniform([self.batch_size, self.z_dim], minval=-1.,
                                                    maxval=1., dtype=tf.float32, name='z2'),
                                reuse=True, batch_size=self.batch_size)
        self.sampler = self.generator(self.sample_z, is_train=False, reuse=True)

        if self.config.dc_discriminator:
            images = self.discriminator(self.images, reuse=False, batch_size=self.real_batch_size)
#            images2 = self.discriminator(self.images2, reuse=True, batch_size=self.real_batch_size)
            G2 = self.discriminator(self.G2, reuse=True)
            G = self.discriminator(self.G, reuse=True)
        else:
            images = tf.reshape(self.images, [self.real_batch_size, -1])
#            images2 = tf.reshape(self.images2, [self.config.test_locations // 2, -1])
            G = tf.reshape(self.G, [self.batch_size, -1])
            G2 = tf.reshape(self.G2, [self.batch_size, -1])

        self.set_loss(G, G2, images)

        block = min(8, int(np.sqrt(self.real_batch_size)), int(np.sqrt(self.batch_size)))
        tf.summary.image("train/input image",
                         self.imageRearrange(tf.clip_by_value(self.images, 0, 1), block))
        tf.summary.image("train/gen image",
                         self.imageRearrange(tf.clip_by_value(self.G, 0, 1), block))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=2)
        
        
    def set_loss(self, G, G2, images):
        bs = min([self.batch_size, self.real_batch_size])
        
        if self.config.single_batch_experiment:
            alpha = tf.constant(np.random.rand(bs), dtype=tf.float32, name='const_alpha')
        else:
            alpha = tf.random_uniform(shape=[bs])
        alpha = tf.reshape(alpha, [bs, 1, 1, 1])
        real_data = self.images[:bs] #before discirminator
        fake_data = self.G[:bs] #before discriminator
        x_hat_data = (1. - alpha) * real_data + alpha * fake_data
        if self.check_numerics:
            x_hat_data = tf.check_numerics(x_hat_data, 'x_hat_data')
        x_hat = self.discriminator(x_hat_data, reuse=True, batch_size=bs)
        
        critic = lambda x, x_ : tf.norm(x - x_, axis=1) - tf.norm(x, axis=1) 
        with tf.variable_scope('loss'):
            if self.config.model == 'deepmind_cramer':
                self.g_loss = tf.reduce_mean(
                    - tf.norm(G - G2) + tf.norm(G - images) + tf.norm(G2 - images))
                self.d_loss = -tf.reduce_mean(critic(images, G) - critic(G2, G))
                to_penalize = critic(x_hat, G)
                
            elif self.config.model == 'reddit_cramer':
                self.g_loss = tf.reduce_mean(critic(images, G) - critic(G, G2))
                d_hat = critic(x_hat, self.x2_)
                self.d_loss = -self.g_loss
                to_penalize = critic(x_hat, G)
                
            elif self.config.model == 'better_cramer':
                S_PQ = tf.reduce_mean(1/2 * tf.norm(G - G2) - tf.norm(G - images))
                self.g_loss = -S_PQ # miminize divergence ~ max expected score S_PQ ~ min -S_PQ
                self.d_loss = S_PQ + tf.norm(images)
                to_penalize = 1/2 * tf.norm(x_hat - G2) - tf.norm(x_hat - images)
                
            elif self.config.model == 'crammer_no_hy':
                S_PQ = tf.reduce_mean(1/2 * tf.norm(G - G2) - tf.norm(G - images))
                self.g_loss = -S_PQ # miminize divergence ~ max expected score S_PQ ~ min -S_PQ
                self.d_loss = S_PQ
                to_penalize = 1/2 * tf.norm(x_hat - G2) - tf.norm(x_hat - images)
                
            gradients = tf.gradients(to_penalize, [x_hat_data])[0]
            if self.check_numerics:
                gradients = tf.check_numerics(gradients, 'gradients 0')
            
            if self.check_numerics:
    #            gradients = tf.check_numerics(tf.clip_by_norm(gradients, 100.), 'gradients F')    
                penalty = tf.check_numerics(tf.reduce_mean(tf.square(tf.norm(gradients, axis=1) - 1.0)), 'penalty')
            else:
    #            gradients = tf.clip_by_norm(gradients, 100.) 
                penalty = tf.reduce_mean(tf.square(tf.norm(gradients, axis=1) - 1.0))#

        
            self.gp = tf.get_variable('gradient_penalty', dtype=tf.float32,
                                      initializer=self.config.gradient_penalty)
            self.d_loss += penalty * self.gp
            self.optim_name = '%s gp %.1f' % (self.config.model, self.config.gradient_penalty)
            # variable_summaries([(gradients, 'dx_gradients')])
            tf.summary.scalar(self.optim_name + ' G', self.g_loss)
            tf.summary.scalar(self.optim_name + ' D', self.d_loss)
            tf.summary.scalar('dx_penalty', penalty)