from .model import MMD_GAN, tf, np
from .architecture import get_networks
from .ops import safer_norm
from . import  mmd

class SMMD_GAN(MMD_GAN):                   
    def set_loss(self, G, images):
        if self.check_numerics:
            G = tf.check_numerics(G, 'G')
            images = tf.check_numerics(images, 'images')
            
        kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
        kerGI = kernel(G, images)
            
        with tf.variable_scope('loss'):
            self.g_loss = mmd.mmd2(kerGI)
            self.d_loss = -self.g_loss 
            self.optim_name = 'smmd_loss'
            
        self.add_gradient_penalty(kernel, G, images)
        self.add_l2_penalty()
        
        print('[*] Loss set')
        
    def add_gradient_penalty(self, kernel, fake, real):
        bs = min([self.batch_size, self.real_batch_size])
        real, fake = real[:bs], fake[:bs]
        
        alpha = tf.random_uniform(shape=[bs, 1, 1, 1])
        real_data = self.images[:bs] # discirminator input level
        fake_data = self.G[:bs] # discriminator input level
        if 'grad_norm' in self.config.suffix:
            if '_PQ' in self.config.suffix:
                x_hat_data = tf.concat([real_data[:bs//2], fake_data[:bs//2]], axis=0)
            elif '_P' in self.config.suffix:
                x_hat_data = real_data
            else:
                x_hat_data = fake_data
        else:
            x_hat_data = (1. - alpha) * real_data + alpha * fake_data
        if self.check_numerics:
            x_hat_data = tf.check_numerics(x_hat_data, 'x_hat_data')
        x_hat = self.discriminator(x_hat_data, bs)
        if self.check_numerics:
            x_hat = tf.check_numerics(x_hat, 'x_hat')
        Ekx = lambda yy: tf.reduce_mean(kernel(x_hat, yy, K_XY_only=True), axis=1)
        Ekxr, Ekxf = Ekx(real), Ekx(fake)
        witness = Ekxr - Ekxf
        if self.check_numerics:
            witness = tf.check_numerics(witness, 'witness')
        gradients = tf.gradients(witness, [x_hat_data])[0]
        if self.check_numerics:
            gradients = tf.check_numerics(gradients, 'gradients 0')

        if 'grad_norm' in self.config.suffix:
            div = 1 + self.gp * tf.reduce_mean(tf.square(safer_norm(gradients, axis=1)))
        else:
            if self.check_numerics:  
                div = 1 + self.gp * tf.check_numerics(tf.reduce_mean(tf.square(safer_norm(gradients, axis=1) - 1.0)), 'penalty')
            else:
                div = 1 + self.gp * tf.reduce_mean(tf.square(safer_norm(gradients, axis=1) - 1.0))

        with tf.variable_scope('loss'):
            if self.config.gradient_penalty > 0:
                self.d_loss /= div
                self.g_loss /= div 
                tf.summary.scalar('div', div)
                print('[*] SMMD denominator added')
            tf.summary.scalar(self.optim_name + ' G', self.g_loss)
            tf.summary.scalar(self.optim_name + ' D', self.d_loss)