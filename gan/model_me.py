import mmd

from model_mmd import DCGAN, tf, np
from utils import variable_summaries
from cholesky import me_loss
from ops import batch_norm, conv2d, deconv2d, linear, lrelu
import time

class me_DCGAN(DCGAN):
    def __init__(self, sess, config, is_crop=True,
                 batch_size=64, output_size=64,
                 z_dim=100, gf_dim=8, df_dim=32,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None, log_dir=None, data_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        super(me_DCGAN, self).__init__(sess=sess, config=config, is_crop=is_crop,
             batch_size=batch_size, output_size=output_size, z_dim=z_dim, 
             gf_dim=gf_dim, df_dim=df_dim, gfc_dim=gfc_dim, dfc_dim=dfc_dim, 
             c_dim=c_dim, dataset_name=dataset_name, checkpoint_dir=checkpoint_dir,
             sample_dir=sample_dir, log_dir=log_dir, data_dir=data_dir)

    def set_loss(self, G, images):
        if self.config.kernel == '':
            me = lambda gg, ii: me_loss(
                gg, ii, self.df_dim, self.batch_size,
                with_inv=True#(self.config.gradient_penalty == 0)
            )
            self.optim_name = 'mean embedding loss'
        else:
            self.me_test_images = tf.placeholder(
                tf.float32, 
                [self.batch_size, self.output_size, self.output_size, self.c_dim],
                name='me_test_locations'
            )
            im_id = tf.constant(np.random.choice(np.arange(self.batch_size), self.df_dim))
            if self.config.dc_discriminator:
                metl = self.discriminator(self.me_test_images, reuse=True)
            else:
                metl = tf.reshape(self.me_test_images, [self.batch_size, -1])
            self.me_test_locations = tf.gather(metl, im_id)
            
            if self.config.kernel == 'rbf': # Gaussian kernel
                bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
                k = lambda gg, ii: mmd._mix_rbf_kernel(
                    gg, ii, sigmas=bandwidths, K_XY_only=True
                )
            elif self.config.kernel == 'rq': # Rational quadratic kernel
                alphas = [.001, .01, .1, 1.0, 10.0]
                k = lambda gg, ii: mmd._mix_rq_kernel(
                    gg, ii, alphas=alphas, K_XY_only=True
                )
            else: 
                raise ValueError("Kernel '%s' not supported" % self.config.kernel)
            me = lambda gg, ii: me_loss(
                k(gg, self.me_test_locations),
                k(ii, self.me_test_locations), 
                self.df_dim, self.batch_size,
                with_inv=(self.config.gradient_penalty == 0)
            )
            self.optim_name = self.config.kernel + ' kernel mean embedding loss'
        with tf.variable_scope('loss'):
            self.optim_loss = me(G, images)
            tf.summary.scalar(self.optim_name, self.optim_loss)

            self.add_gradient_penalty(me, G, images)      

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

#            if True: #np.mod(s, 16) == 0:
##                h0 = self.d_bn0(image)
##                h0 = h0 + lrelu(conv2d(h0, self.c_dim, name='d_h0_conv', d_h=1, d_w=1))
##                h1 = self.d_bn1(h0, train=True)
##                h1 = h1 + lrelu(conv2d(h1, self.c_dim, name='d_h1_conv', d_h=1, d_w=1))
##                h2 = self.d_bn2(h1, train=True)
##                h2 = h2 + lrelu(conv2d(h2, self.c_dim, name='d_h2_conv', d_h=1, d_w=1))
##                h3 = self.d_bn3(h2, train=True)
##                h3 = h3 + lrelu(conv2d(h3, self.c_dim, name='d_h3_conv', d_h=1, d_w=1))
#                return linear(tf.reshape(image, [self.batch_size, -1]), self.df_dim, 'd_h4_lin')
            
            s = self.df_dim
            ch = np.ceil(self.output_size/16) ** 2
            s0, s1, s2, s3 = max(1, int(s/(ch*8))), max(1, int(s/(ch*4))), \
                        max(1, int(s/(ch*2))), max(1, int(s/ch))
            h0 = lrelu(self.d_bn0(conv2d(image, s0, name='d_h0_conv')))
            h1 = lrelu(self.d_bn1(conv2d(h0, s1, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, s2, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, s3, name='d_h3_conv')))
#            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), self.df_dim, 'd_h3_lin')
            return tf.reshape(h3, [self.batch_size, -1])

    def train_step(self, config, batch_images):
        batch_z = np.random.uniform(
            -1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

        if self.config.use_kernel:
            feed_dict = {self.lr: self.current_lr, self.images: batch_images,
                         self.z: batch_z}
            if self.config.kernel != '':
                feed_dict.update({self.me_test_images: self.additional_sample_images})
            if self.config.is_demo:
                summary_str, step, optim_loss = self.sess.run(
                    [self.TrainSummary, self.global_step, self.optim_loss],
                    feed_dict=feed_dict
                )
            else:
                if self.d_counter == 0:
                    _, summary_str, step, optim_loss = self.sess.run(
                        [self.g_grads, self.TrainSummary, self.global_step,
                         self.optim_loss], feed_dict=feed_dict
                    )
                else:
    #                        (np.mod(counter//100, 5) == 4) and \
    #                        (counter < self.config.max_iteration * 4/4):
                    _, summary_str, step, optim_loss = self.sess.run(
                        [self.d_grads, self.TrainSummary, self.global_step,
                         self.optim_loss], feed_dict=feed_dict
                    )     
        # G STEP
        if self.d_counter == 0:
            if (np.mod(self.counter, 10) == 1) or (self.err_counter > 0):
                try:
                    self.writer.add_summary(summary_str, step)
                    self.err_counter = 0
                except Exception as e:
                    print('Step %d summary exception. ' % self.counter, e)
                    self.err_counter += 1
                    
                print("Epoch: [%2d] time: %4.4f, %s: %.8f"
                    % (self.counter, time.time() - self.start_time, self.optim_name, optim_loss)) 
            if (np.mod(self.counter, self.config.max_iteration//5) == 0):
                self.current_lr *= self.config.decay_rate
                print('current learning rate: %f' % self.current_lr)  
            
        if self.counter == 1:
            print('current learning rate: %f' % self.current_lr)
        if self.config.dc_discriminator:
            d_steps = 2
            if (self.counter % 100 == 0) or (self.counter < 20):
                d_steps = 100
            self.d_counter = (self.d_counter + 1) % (d_steps + 1)
        self.counter += (self.d_counter == 0)
        
        return summary_str, step, optim_loss