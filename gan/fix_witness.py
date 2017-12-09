import mmd as MMD
from mmd import _eps, _check_numerics

from model_mmd2 import MMD_GAN, tf, np
from model_me_brb import MEbrb_GAN
from utils import variable_summaries, safer_norm
from cholesky import me_loss
from ops import batch_norm, conv2d, deconv2d, linear, lrelu
from glob import glob
import os
import time

class FixWitness(MMD_GAN): 
    def set_loss(self, G, images):
        bs = min([self.batch_size, self.real_batch_size])
        
        if self.config.single_batch_experiment:
            alpha = tf.constant(np.random.rand(bs), dtype=tf.float32, name='const_alpha')
        else:
            alpha = tf.random_uniform(shape=[bs])
        if 'mid' in self.config.suffix:
            alpha = .4 + .2 * alpha
        elif 'edges' in self.config.suffix:
            qq = tf.cast(tf.reshape(tf.multinomial([[.5, .5]], bs),
                                    [bs]), tf.float32)
            alpha = .1 * alpha * qq + (1. - .1 * alpha) * (1. - qq)
        elif 'edge' in self.config.suffix:
            alpha = .95 + .05 * alpha
        alpha = tf.reshape(alpha, [bs, 1, 1, 1])
        real_data = self.images[: bs] #before discirminator
        fake_data = self.G[: bs] #before discriminator
        x_hat_data = (1. - alpha) * real_data + alpha * fake_data
        if self.check_numerics:
            x_hat_data = tf.check_numerics(x_hat_data, 'x_hat_data')
        x_hat = self.discriminator(x_hat_data, reuse=True, batch_size=bs)        
        
        if self.check_numerics:
            G = tf.check_numerics(G, 'G')
            images = tf.check_numerics(images, 'images')
        if self.config.kernel == 'di': # Distance - induced kernel
            self.di_kernel_z_images = tf.constant(
                self.additional_sample_images,
                dtype=tf.float32,                                  
                name='di_kernel_z_images'
            )
            alphas = [1.0]
            di_r = np.random.choice(np.arange(self.batch_size))
            if self.config.dc_discriminator:
                self.di_kernel_z = self.discriminator(
                        self.di_kernel_z_images, reuse=True)[di_r: di_r + 1]
            else:
                self.di_kernel_z = tf.reshape(self.di_kernel_z_images[di_r: di_r + 1], [1, -1])
            kernel = lambda gg, ii, K_XY_only=False: MMD._mix_di_kernel(
                    gg, ii, self.di_kernel_z, alphas=alphas, K_XY_only=K_XY_only)
        else:
            kernel = getattr(MMD, '_%s_kernel' % self.config.kernel)
            
        witness_G = tf.Variable(np.random.rand(self.batch_size, self.dof_dim), 
                                name='witness_G', trainable=False, dtype=tf.float32)
        witness_I = tf.Variable(np.random.rand(self.batch_size, self.dof_dim), 
                                name='witness_I', trainable=False, dtype=tf.float32)
        
        self.assign_witness_op = tf.group(tf.assign(witness_G, self.d_G),
                                          tf.assign(witness_I, self.d_images))
        
        witness = lambda z: tf.reduce_mean(kernel(witness_I, z, K_XY_only=True) - 
                                           kernel(witness_G, z, K_XY_only=True))
        
        gradients = tf.gradients(x_hat, [x_hat_data])[0]
        if self.check_numerics:  
            penalty = tf.check_numerics(tf.reduce_mean(tf.square(safer_norm(gradients, axis=1) - 1.0)), 'penalty')
        else:
            penalty = tf.reduce_mean(tf.square(safer_norm(gradients, axis=1) - 1.0))
        
        with tf.variable_scope('loss'):
            self.g_loss= -witness(G)
            self.d_loss= -witness(images) + witness(G) + penalty * self.gp
            
            self.optim_name = 'fix witness mmd gp %.1f' % self.config.gradient_penalty
            tf.summary.scalar('dx_penalty', penalty)
            print('[*] Gradient penalty added')
            tf.summary.scalar(self.optim_name + ' G', self.g_loss)
            tf.summary.scalar(self.optim_name + ' D', self.d_loss)
            
        self.add_l2_penalty()
        
        print('[*] Loss set')     
        
    def train_step(self, batch_images=None):
        step = self.sess.run(self.global_step)
        if step % self.config.witness_update_frequency == 0:
            self.sess.run(self.assign_witness_op)
        
        return super(FixWitness, self).train_step(batch_images=batch_images)