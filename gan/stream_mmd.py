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

def rbf_mmd2_streaming(X, Y, log_sigma=0):
   # n = (T.smallest(X.shape[0], Y.shape[0]) // 2) * 2
   n = (X.shape[0] // 2) * 2
   gamma = 1 / (2 * np.exp(2 * log_sigma))
   rbf = lambda A, B: T.exp(-gamma * ((A - B) ** 2).sum(axis=1))
   mmd2 = (rbf(X[:n:2], X[1:n:2]) + rbf(Y[:n:2], Y[1:n:2])
         - rbf(X[:n:2], Y[1:n:2]) - rbf(X[1:n:2], Y[:n:2])).mean()
   return mmd2, mmd2
   
def _distance(X, Y):
    return safer_norm(X, axis=1) + safer_norm(Y, axis=1) - safer_norm(X - Y, axis=1) 
    
def _mix_rq(X, Y, alphas=[.1, 1., 10.], wts=None, check_numerics=_check_numerics):
    """
    Rational quadratic kernel
    http://www.cs.toronto.edu/~duvenaud/cookbook/index.html
    """
    if wts is None:
        wts = [1.] * len(alphas)
    
    K_XY = 0.
    XYsqnorm = tf.reduce_mean((X - Y)**2, axis=1)
    for alpha, wt in zip(alphas, wts):
        logXY = tf.log(1. + XYsqnorm/(2.*alpha))
        if check_numerics:
            logXY = tf.check_numerics(logXY, 'K_XY_log %f' % alpha)
        K_XY += wt * tf.exp(-alpha * logXY)
    return K_XY
    

class Stream_MMD_GAN(MMD_GAN): 
    def set_loss(self, G, images):
        if self.check_numerics:
            G = tf.check_numerics(G, 'G')
            images = tf.check_numerics(images, 'images')
            
        bs = min([self.batch_size, self.real_batch_size])
        bs2 = self.batch_size//2
        
        if self.config.single_batch_experiment:
            alpha = tf.constant(np.random.rand(bs2), dtype=tf.float32, name='const_alpha')
        else:
            alpha = tf.random_uniform(shape=[bs2])
        alpha = tf.reshape(alpha, [bs2, 1, 1, 1])
        real_data = self.images[bs2: 2 * bs2] #before discirminator
        fake_data = self.G[bs2: 2 * bs2] #before discriminator
        x_hat_data = (1. - alpha) * real_data + alpha * fake_data
        if self.check_numerics:
            x_hat_data = tf.check_numerics(x_hat_data, 'x_hat_data')
        x_hat = self.discriminator(x_hat_data, reuse=True, batch_size=bs2)
        
        with tf.variable_scope('loss'):
            if self.config.kernel == 'distance':
                kernel = _distance
            elif self.config.kernel == 'mix_rq':
                kernel = _mix_rq
            
            G1, G2 = G[:bs2], G[bs2: 2 * bs2]
            I1, I2 = images[:bs2], images[bs2: 2 * bs2]

            witness = lambda z: tf.reduce_mean(kernel(G1, z) - kernel(I1, z))

            self.g_loss = witness(G2) - witness(I2)
            self.d_loss = -self.g_loss
            
            to_penalize = witness(x_hat)
            gradients = tf.gradients(to_penalize, [x_hat_data])[0]
            if self.check_numerics:
                gradients = tf.check_numerics(gradients, 'gradients 0')
            if self.check_numerics:  
                penalty = tf.check_numerics(tf.reduce_mean(tf.square(safer_norm(gradients, axis=1) - 1.0)), 'penalty')
            else:
                penalty = tf.reduce_mean(tf.square(safer_norm(gradients, axis=1) - 1.0))#

        
            self.gp = tf.get_variable('gradient_penalty', dtype=tf.float32,
                                      initializer=self.config.gradient_penalty)
            self.d_loss += penalty * self.gp
            
            self.optim_name = '%s gp %.1f' % (self.config.model, self.config.gradient_penalty)
            tf.summary.scalar(self.optim_name + ' G', self.g_loss)
            tf.summary.scalar(self.optim_name + ' D', self.d_loss)
            tf.summary.scalar('dx_penalty', penalty)
            self.add_l2_penalty()