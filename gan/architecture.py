#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:34:47 2018

@author: mikolajbinkowski
"""
import tensorflow as tf
from ops import batch_norm, conv2d, deconv2d, linear, lrelu
from utils import conv_sizes, variable_summaries
from mmd import _debug
# Generators

class Generator:
    def __init__(self, dim, c_dim, output_size, use_batch_norm, prefix='g_'):
        self.used = False
        self.dim = dim
        self.c_dim = c_dim
        self.output_size = output_size
        self.prefix = prefix
        if use_batch_norm:
            self.g_bn0 = batch_norm(name=prefix + 'bn0')
            self.g_bn1 = batch_norm(name=prefix + 'bn1')
            self.g_bn2 = batch_norm(name=prefix + 'bn2')
            self.g_bn3 = batch_norm(name=prefix + 'bn3')
            self.g_bn4 = batch_norm(name=prefix + 'bn4')
            self.g_bn5 = batch_norm(name=prefix + 'bn5')
        else:
            self.g_bn0 = lambda x: x
            self.g_bn1 = lambda x: x
            self.g_bn2 = lambda x: x
            self.g_bn3 = lambda x: x
            self.g_bn4 = lambda x: x
            self.g_bn5 = lambda x: x
            
    def __call__(self, seed, batch_size):
        with tf.variable_scope('generator') as scope:   
            if self.used:
                scope.reuse_variables()
            self.used = True
            return self.network(seed, batch_size)
        
    def network(self, seed, batch_size):
        pass

    
class DCGANGenerator(Generator):
    def network(self, seed, batch_size):
        s1, s2, s4, s8, s16 = conv_sizes(self.output_size, layers=4, stride=2)
        # 64, 32, 16, 8, 4 - for self.output_size = 64
        # default architecture
        # For Cramer: self.gf_dim = 64
        z_ = linear(seed, self.dim * 8 * s16 * s16, self.prefix + 'h0_lin') # project random noise seed and reshape
        
        h0 = tf.reshape(z_, [batch_size, s16, s16, self.dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0))
        
        h1 = deconv2d(h0, [batch_size, s8, s8, self.dim*4], name=self.prefix + 'h1')
        h1 = tf.nn.relu(self.g_bn1(h1))
                        
        h2 = deconv2d(h1, [batch_size, s4, s4, self.dim*2], name=self.prefix + 'h2')
        h2 = tf.nn.relu(self.g_bn2(h2))
        
        h3 = deconv2d(h2, [batch_size, s2, s2, self.dim*1], name=self.prefix + 'h3')
        h3 = tf.nn.relu(self.g_bn3(h3))
        
        h4 = deconv2d(h3, [batch_size, s1, s1, self.c_dim], name=self.prefix + 'h4')
        return tf.nn.sigmoid(h4)        


class ResNetGenerator(Generator):
    def network(self, seed, batch_size):
        from resnet import ResidualBlock
        import tflib as lib
        s1, s2, s4, s8, s16, s32 = conv_sizes(self.output_size, layers=5, stride=2)
        # project `z` and reshape
        z_= linear(seed, self.dim * 16 * s32 * s32, self.prefix + 'h0_lin')
        h0 = tf.reshape(z_, [-1, self.dim * 16, s32, s32]) # NCHW format
        h1 = ResidualBlock(self.prefix + 'res1', 16 * self.dim, 
                           8 * self.dim, 3, h0, resample='up')
        h2 = ResidualBlock(self.prefix + 'res2', 8 * self.dim, 
                           4 * self.dim, 3, h1, resample='up')
        h3 = ResidualBlock(self.prefix + 'res3', 4 * self.dim, 
                           2 * self.dim, 3, h2, resample='up')
        h4 = ResidualBlock(self.prefix + 'res4', 2 * self.dim, 
                           self.dim, 3, h3, resample='up')
        h4 = lib.ops.batchnorm.Batchnorm('g_h4', [0, 2, 3], h4, fused=True)
        h4 = tf.nn.relu(h4)
#                h5 = lib.ops.conv2d.Conv2D('g_h5', dim, 3, 3, h4)
        h5 = tf.transpose(h4, [0, 2, 3, 1]) # NCHW to NHWC
        h5 = deconv2d(h5, [batch_size, s1, s1, self.c_dim], name='g_h5')
        return tf.nn.sigmoid(h5)


# Discriminator

class Discriminator:
    def __init__(self, dim, o_dim, use_batch_norm, prefix='d_'):
        self.dim = dim
        self.o_dim = o_dim if (o_dim > 0) else 8 * dim
        self.prefix = prefix
        self.used = False
        if use_batch_norm:
            self.d_bn0 = batch_norm(name=prefix + 'bn0')
            self.d_bn1 = batch_norm(name=prefix + 'bn1')
            self.d_bn2 = batch_norm(name=prefix + 'bn2')
            self.d_bn3 = batch_norm(name=prefix + 'bn3')
            self.d_bn4 = batch_norm(name=prefix + 'bn4')
            self.d_bn5 = batch_norm(name=prefix + 'bn5')
        else:
            self.d_bn0 = lambda x: x
            self.d_bn1 = lambda x: x
            self.d_bn2 = lambda x: x
            self.d_bn3 = lambda x: x
            self.d_bn4 = lambda x: x
            self.d_bn5 = lambda x: x
        
    def __call__(self, image, batch_size, return_layers=False):
        with tf.variable_scope("discriminator") as scope:
            if self.used:
                scope.reuse_variables()
            self.used = True
            
            layers = self.network(image, batch_size)
            
            if _debug:
                variable_summaries(layers)
            if return_layers:
                return layers
            return layers['hF']
        
    def network(self, image, batch_size):
        pass

class DCGANDiscriminator(Discriminator):        
    def network(self, image, batch_size):
        h0 = lrelu(conv2d(image, self.dim, name=self.prefix + 'h0_conv')) 
        h1 = lrelu(self.d_bn1(conv2d(h0, self.dim * 2, name=self.prefix + 'h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.dim * 4, name=self.prefix + 'h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.dim * 8, name=self.prefix + 'h3_conv')))
        hF = linear(tf.reshape(h3, [batch_size, -1]), self.o_dim, self.prefix + 'h4_lin')
        
        return {'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'hF': hF}

        
def get_networks(architecture):
    if 'dcgan' in architecture:
        return DCGANGenerator, DCGANDiscriminator
    elif 'resnet5' in architecture:
        return ResNetGenerator, DCGANDiscriminator

#    def generator(self, z, y=None, is_train=True, reuse=False, batch_size=None):
#        if batch_size is None:
#            batch_size = self.batch_size
#        if self.dataset not in ['mnist', 'cifar10', 'lsun', 'GaussianMix', 'celebA']:
#            raise ValueError("not implemented dataset '%s'" % self.dataset)
#        elif self.dataset in ['lsun', 'cifar10']:
#            if self.config.architecture == 'mlp':
#                return self.MLP_generator(z, is_train=is_train, reuse=reuse)
#        with tf.variable_scope('generator') as scope:
#            if reuse:
#                scope.reuse_variables()
#            s1, s2, s4, s8, s16 = conv_sizes(self.output_size, layers=4, stride=2)
#            # 64, 32, 16, 8, 4 - for self.output_size = 64
#            if 'dcgan' in self.config.architecture:
#                # default architecture
#                # For Cramer: self.gf_dim = 64
#                z_ = linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin') # project random noise seed and reshape
#                
#                h0 = tf.reshape(z_, [batch_size, s16, s16, self.gf_dim * 8])
#                h0 = tf.nn.relu(self.g_bn0(h0))
#                
#                h1 = deconv2d(h0, [batch_size, s8, s8, self.gf_dim*4], name='g_h1')
#                h1 = tf.nn.relu(self.g_bn1(h1))
#                                
#                h2 = deconv2d(h1, [batch_size, s4, s4, self.gf_dim*2], name='g_h2')
#                h2 = tf.nn.relu(self.g_bn2(h2))
#                
#                h3 = deconv2d(h2, [batch_size, s2, s2, self.gf_dim*1], name='g_h3')
#                h3 = tf.nn.relu(self.g_bn3(h3))
#                
#                h4 = deconv2d(h3, [batch_size, s1, s1, self.c_dim], name='g_h4')
#                return tf.nn.sigmoid(h4)
#            
#            elif 'dfc' in self.config.architecture:
#                z_ = tf.reshape(z, [batch_size, 1, 1, -1])
#                h0 = tf.nn.relu(self.g_bn0(deconv2d(z_, 
#                    [batch_size, s8, s8, self.gf_dim * 4], name='g_h0_conv',
#                    k_h=4, k_w=4, d_h=4, d_w=4)))
#                h1 = tf.nn.relu(self.g_bn1(deconv2d(h0, 
#                    [batch_size, s4, s4, self.gf_dim * 2], name='g_h1_conv', k_h=4, k_w=4)))
#                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, 
#                    [batch_size, s2, s2, self.gf_dim], name='g_h2_conv', k_h=4, k_w=4)))
#                h3 = deconv2d(h2, [batch_size, s1, s1, self.c_dim], name='g_h3_conv', k_h=4, k_w=4)
#                return tf.nn.sigmoid(h3)
#            
#            elif 'dcold' in self.config.architecture:
#                # project `z` and reshape
#                self.z_, self.h0_w, self.h0_b = linear(
#                    z, self.gf_dim*8*s16*s16, 'g_h0_lin', with_w=True)
#
#                h0 = tf.nn.relu(self.g_bn0(self.z_))
#                
#                h1, self.h1_w, self.h1_b = linear(
#                        h0, self.gf_dim*4*s8*s8, 'g_h1_lin', with_w=True)
#                h1 = tf.nn.relu(self.g_bn1(tf.reshape(h1, [-1, s8, s8, self.gf_dim*4])))
#                
#                h2, self.h2_w, self.h2_b = deconv2d(
#                    h1, [batch_size, s4, s4, self.gf_dim*2], name='g_h2', with_w=True)
#                h2 = tf.nn.relu(self.g_bn2(h2))
#                
#                h3, self.h3_w, self.h3_b = deconv2d(
#                    h2, [batch_size, s2, s2, self.gf_dim*1], name='g_h3', with_w=True)
#                h3 = tf.nn.relu(self.g_bn3(h3))
#                
#                h4, self.h4_w, self.h4_b = deconv2d(
#                    h3, [batch_size, s1, s1, self.c_dim], name='g_h4', with_w=True)
#                return tf.nn.sigmoid(h4)
#            
#            elif 'dc64' in self.config.architecture:
#                s1, s2, s4, s8, s16, s32 = conv_sizes(self.output_size, layers=5, stride=2)
#                # project `z` and reshape
#                z_= linear(z, self.gf_dim*16*s32*s32, 'g_h0_lin')
#                
#                h0 = tf.reshape(z_, [-1, s32, s32, self.gf_dim * 16])
#                h0 = tf.nn.relu(self.g_bn0(h0))
#                
#                h1 = deconv2d(h0, [batch_size, s16, s16, self.gf_dim*8], name='g_h1')
#                h1 = tf.nn.relu(self.g_bn1(h1))
#                                
#                h2 = deconv2d(h1, [batch_size, s8, s8, self.gf_dim*4], name='g_h2')
#                h2 = tf.nn.relu(self.g_bn2(h2))
#
#                h3 = deconv2d(h2, [batch_size, s4, s4, self.gf_dim*2], name='g_h3')
#                h3 = tf.nn.relu(self.g_bn3(h3))
#
#                h4 = deconv2d(h3, [batch_size, s2, s2, self.gf_dim], name='g_h4')
#                h4 = tf.nn.relu(self.g_bn4(h4))                
#                
#                h5 = deconv2d(h4, [batch_size, s1, s1, self.c_dim], name='g_h5')
#                return tf.nn.sigmoid(h5)
#            
#            elif 'dc128' in self.config.architecture:
#                s1, s2, s4, s8, s16, s32, s64 = conv_sizes(self.output_size, layers=6, stride=2)
#                # project `z` and reshape
#                z_= linear(z, self.gf_dim*32*s64*s64, 'g_h0_lin')
#
#                h0 = tf.reshape(z_, [-1, s64, s64, self.gf_dim * 32])
#                h0 = tf.nn.relu(self.g_bn0(h0))
#
#                h1 = deconv2d(h0, [batch_size, s32, s32, self.gf_dim*16], name='g_h1')
#                h1 = tf.nn.relu(self.g_bn1(h1))
#
#                h2 = deconv2d(h1, [batch_size, s16, s16, self.gf_dim*8], name='g_h2')
#                h2 = tf.nn.relu(self.g_bn2(h2))
#
#                h3 = deconv2d(h2, [batch_size, s8, s8, self.gf_dim*4], name='g_h3')
#                h3 = tf.nn.relu(self.g_bn3(h3))
#
#                h4 = deconv2d(h3, [batch_size, s4, s4, self.gf_dim*2], name='g_h4')
#                h4 = tf.nn.relu(self.g_bn4(h4))
#
#                h5 = deconv2d(h4, [batch_size, s2, s2, self.gf_dim*1], name='g_h5')
#                h5 = tf.nn.relu(self.g_bn5(h5))
#
#                h6 = deconv2d(h5, [batch_size, s1, s1, self.c_dim], name='g_h6')
#                return tf.nn.sigmoid(h6)
#            elif 'resnet5' in self.config.architecture:
#                from resnet import ResidualBlock
#                import tflib as lib
#                s1, s2, s4, s8, s16, s32 = conv_sizes(self.output_size, layers=5, stride=2)
#                # project `z` and reshape
#                z_= linear(z, self.gf_dim*16*s32*s32, 'g_h0_lin')
#                dim = self.gf_dim
#                h0 = tf.reshape(z_, [-1, dim * 16, s32, s32]) # NCHW format
#                h1 = ResidualBlock('g_res1',16*dim, 8*dim, 3, h0, resample='up')
#                h2 = ResidualBlock('g_res2', 8*dim, 4*dim, 3, h1, resample='up')
#                h3 = ResidualBlock('g_res3', 4*dim, 2*dim, 3, h2, resample='up')
#                h4 = ResidualBlock('g_res4', 2*dim, 1*dim, 3, h3, resample='up')
#                h4 = lib.ops.batchnorm.Batchnorm('g_h4', [0, 2, 3], h4, fused=True)
#                h4 = tf.nn.relu(h4)
##                h5 = lib.ops.conv2d.Conv2D('g_h5', dim, 3, 3, h4)
#                h5 = tf.transpose(h4, [0, 2, 3, 1]) # NCHW to NHWC
#                h5 = deconv2d(h5, [batch_size, s1, s1, self.c_dim], name='g_h5')
#                return tf.nn.sigmoid(h5)

#    def discriminator(self, image, y=None, reuse=False, batch_size=None, 
#                      return_layers=False):
#        if batch_size is None:
#            batch_size = self.batch_size
#        with tf.variable_scope("discriminator") as scope:
#            layers = {}
#            if reuse:
#                scope.reuse_variables()
#            if ('dcgan' in self.config.architecture): # default architecture
#                if self.dof_dim <= 0:
#                    self.dof_dim = self.df_dim * 8
#                # For Cramer:
#                # self.dof_dim = 256
#                # self.df_dim = 64
#                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv')) 
#                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
#                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
#                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
#                hF = linear(tf.reshape(h3, [batch_size, -1]), self.dof_dim, 'd_h4_lin')
#            elif 'dfc' in self.config.architecture:
#                h0 = lrelu(conv2d(image, self.df_dim, k_h=4, k_w=4, name='d_h0_conv'))
#                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, k_h=4, k_w=4, name='d_h1_conv')))
#                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, k_h=4, k_w=4, name='d_h2_conv')))
#                h3 = conv2d(h2, self.df_dim, d_h=4, d_w=4, k_h=4, k_w=4, name='d_h3_conv')
#                hF = tf.reshape(h3, [batch_size, self.df_dim])
#                self.dof_dim = self.df_dim
#            elif 'dcold' in self.config.architecture:
#                h0 = lrelu(conv2d(image, self.df_dim//8, name='d_h0_conv'))
#                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim//4, name='d_h1_conv')))
#                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim//2, name='d_h2_conv')))
#                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim, name='d_h3_conv')))
#                hF = linear(tf.reshape(h3, [batch_size, -1]), self.df_dim, 'd_h4_lin')
#                self.dof_dim = self.df_dim
#            elif ('dc64' in self.config.architecture)  or ('g-resnet5' in self.config.architecture):
#                if self.dof_dim <= 0:
#                    self.dof_dim = self.df_dim * 16
#                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
#                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
#                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
#                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
#                h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim * 16, name='d_h4_conv')))
#                hF = linear(tf.reshape(h4, [batch_size, -1]), self.dof_dim, 'd_h6_lin')
#                layers['h4'] = h4
#            elif 'dc128' in self.config.architecture:
#                if self.dof_dim <= 0:
#                    self.dof_dim = self.df_dim * 32
#                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
#                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
#                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
#                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
#                h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim * 16, name='d_h4_conv')))
#                h5 = lrelu(self.d_bn5(conv2d(h4, self.df_dim * 32, name='d_h5_conv')))
#                hF = linear(tf.reshape(h5, [batch_size, -1]), self.dof_dim , 'd_h6_lin')
#                layers.update({'h4': h4, 'h5': h5})
#            else:
#                raise ValueError("Choose architecture from  [dfc, dcold, dcgan, dc64, dc128]")
#            print(repr(image.get_shape()).replace('Dimension', '') + ' --> Discriminator --> ' + \
#                  repr(hF.get_shape()).replace('Dimension', ''))
#            
#            layers.update({'h0': h0, 'h1': h1, 'h2': h2, 'h3': h3, 'hF': hF})
#            
#            if _debug:
#                variable_summaries(layers)
#            
#            if return_layers:
#                return layers
#                
#            return hF