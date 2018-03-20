from .model import MMD_GAN, tf
from . import  mmd
from .ops import safer_norm, tf

class WMMD(MMD_GAN):
    def __init__(self, sess, config, **kwargs):
        config.dof_dim = 1
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
        x_hat_data = self.images[:bs]
        x_hat = self.discriminator(x_hat_data, bs)
        if self.config.d_is_injective:
            grad_x_hat = tf.gradients(x_hat[:,:-self.input_dim ], [x_hat_data])
            scale = tf.reduce_mean( tf.reduce_sum( tf.square(grad_x_hat), axis = [1,2,3])  + tf.square(self.discriminator.scale_id_layer)*self.input_dim )
        else:
            grad_x_hat = tf.gradients(x_hat, [x_hat_data])
            scale = tf.reduce_mean( tf.reduce_sum( tf.square(grad_x_hat), axis = [1,2,3]) )
        unscaled_g_loss = 1*self.g_loss
        with tf.variable_scope('loss'):
            if self.config.hessian_scale:
                if self.config.d_is_injective:
                    tf.summary.scalar(self.optim_name + 'id_scale', tf.reduce_mean(self.discriminator.scale_id_layer))
                tf.summary.scalar(self.optim_name + '_unscaled_G', unscaled_g_loss)
                self.apply_scaling(scale)
                tf.summary.scalar('dx_scale', scale)
                print('[*] Hessian Scaling added')
                tf.summary.scalar(self.optim_name + '_G', self.g_loss)
                tf.summary.scalar(self.optim_name + '_D', self.d_loss)
    def apply_scaling(self,scale):
        if self.config.scale_variant == 0:
            self.d_loss /= scale
            self.g_loss /= scale
        elif self.config.scale_variant == 1:
            self.d_loss /= scale
        elif self.config.scale_variant == 2:
            self.d_loss /= (self.hs*scale+1)
            self.g_loss /= (self.hs*scale+1)
        elif self.config.scale_variant == 3:
            self.d_loss /= tf.maximum(self.hs*scale+1, 4.)
            self.g_loss /= tf.maximum(self.hs*scale+1, 4.)
        print('[*] Adding scale variant %d', self.config.scale_variant)










