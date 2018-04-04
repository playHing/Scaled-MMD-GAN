from .model import MMD_GAN, tf
from . import  mmd
from .ops import safer_norm, tf, squared_norm_jacobian

class WMMD(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        #config.dof_dim = 1
        super(WMMD, self).__init__(sess, config, **kwargs)
        
    def set_loss(self, G, images):
        kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
        kerGI = kernel(G, images)
            
        with tf.variable_scope('loss'):
            self.g_loss = mmd.mmd2(kerGI)
            self.d_loss = -self.g_loss 
            self.optim_name = 'kernel_loss'
        self.scale_by_hs_norm(kernel)
        self.add_gradient_penalty(kernel, G, images)
        self.add_l2_penalty()

        print('[*] Loss set')
    def scale_by_hs_norm(self,kernel):


        bs = min([self.batch_size, self.real_batch_size])
        #alpha = tf.random_uniform(shape=[bs, 1, 1, 1])
        if self.config.grad_expectation == 0:
            x_hat_data = self.images[:bs]
        elif self.config.grad_expectation == 1:
            x_hat_data = self.G[:bs]
        elif self.config.grad_expectation == 2:
            alpha = tf.random_uniform(shape=[bs, 1, 1, 1])
            x_hat_data =  (1. - alpha) * self.images[:bs] + alpha * self.G[:bs]
        elif self.config.grad_expectation == 3:
            alpha = tf.random_uniform(shape=[bs, 1, 1, 1])
            x_hat_data =  tf.boolean_mask(self.images[:bs],(alpha<0.5))  + tf.boolean_mask(self.G[:bs],(alpha>=0.5))
        x_hat_data = tf.stop_gradient(x_hat_data)
        x_hat = self.discriminator(x_hat_data, bs)
        if self.config.d_is_injective:
            norme2_jac = squared_norm_jacobian(x_hat[:,:-self.input_dim ], x_hat_data)
            avg_norme2_jac = tf.reduce_mean( norme2_jac  + tf.square(self.discriminator.scale_id_layer)*self.input_dim )
        else:
            norme2_jac = squared_norm_jacobian(x_hat, x_hat_data)
            avg_norme2_jac = tf.reduce_mean( norme2_jac )
        self.unscaled_g_loss = 1.*self.g_loss
        self.unscaled_d_loss = 1.*self.d_loss
        self.norm_discriminator = tf.reduce_mean(tf.square( x_hat ))
        with tf.variable_scope('loss'):
            if self.config.hessian_scale:
                if self.config.d_is_injective:
                    tf.summary.scalar(self.optim_name + 'id_scale', tf.reduce_mean(self.discriminator.scale_id_layer))
                tf.summary.scalar(self.optim_name + '_unscaled_G', self.unscaled_g_loss)
                self.apply_scaling(avg_norme2_jac,self.norm_discriminator)
                tf.summary.scalar('dx_scale', avg_norme2_jac)
                print('[*] Hessian Scaling added')
                tf.summary.scalar(self.optim_name + '_G', self.g_loss)
                tf.summary.scalar(self.optim_name + '_D', self.d_loss)
                tf.summary.scalar(self.optim_name + '_norm_D', self.norm_discriminator)
    def apply_scaling(self,scale,norm):
        if self.config.scale_variant == 0:
            epsilon = 0.00000001
            self.scale= (self.hs*scale + epsilon)

        elif self.config.scale_variant == 2:
            self.scale= 1./(self.hs*scale+1.)
        elif self.config.scale_variant == 3:
            self.scale= 1./(tf.maximum(self.hs*scale+1, 4.))
        elif self.config.scale_variant == 4:
            epsilon = 0.00000001
            self.scale= 1+1./(self.hs*scale+epsilon)
        elif self.config.scale_variant == 6:
            self.scale= 1+1./(self.hs*scale+1)
        elif self.config.scale_variant == 8:
            self.scale= 1./(self.hs*(scale + norm) +1)

        self.g_loss = self.g_loss*self.scale
        self.d_loss = -self.g_loss
        print('[*] Adding scale variant %d', self.config.scale_variant)





