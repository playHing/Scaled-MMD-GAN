import mmd as MMD
from mmd import _eps

from model_mmd2 import MMD_GAN, tf, np
from model_me2 import ME_GAN
from utils import variable_summaries
from cholesky import me_loss
from ops import batch_norm, conv2d, deconv2d, linear, lrelu
from glob import glob
import os
import time

class MEbrb_GAN(ME_GAN):
    def build_model(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.get_variable('lr', dtype=tf.float32, initializer=self.config.learning_rate)
        if 'lsun' in self.config.dataset:
            self.set_lmdb_pipeline(streams=[self.real_batch_size, self.config.test_locations // 2])
        else:
            self.set_input_pipeline(streams=[self.real_batch_size, self.config.test_locations // 2])


        self.sample_z = tf.constant(np.random.uniform(-1, 1, size=(self.sample_size,
                                                      self.z_dim)).astype(np.float32),
                                    dtype=tf.float32, name='sample_z')

        self.G = self.generator(tf.random_uniform([self.batch_size, self.z_dim], minval=-1.,
                                                   maxval=1., dtype=tf.float32, name='z'))
        self.G2 = self.generator(tf.random_uniform([self.config.test_locations // 2, self.z_dim], minval=-1.,
                                                    maxval=1., dtype=tf.float32, name='z2'),
                            reuse=True, batch_size=self.config.test_locations // 2)
        self.sampler = self.generator(self.sample_z, is_train=False, reuse=True)

        if self.config.dc_discriminator:
            images = self.discriminator(self.images, reuse=False, batch_size=self.real_batch_size)
            images2 = self.discriminator(self.images2, reuse=True, batch_size=self.config.test_locations // 2)
            G2 = self.discriminator(self.G2, reuse=True, batch_size=self.config.test_locations // 2)
            G = self.discriminator(self.G, reuse=True)
        else:
            images = tf.reshape(self.images, [self.real_batch_size, -1])
            images2 = tf.reshape(self.images2, [self.config.test_locations // 2, -1])
            G = tf.reshape(self.G, [self.batch_size, -1])
            G2 = tf.reshape(self.G2, [self.config.test_locations // 2, -1])

        self.set_loss(G, G2, images, images2)

        block = min(8, int(np.sqrt(self.real_batch_size)), int(np.sqrt(self.batch_size)))
        tf.summary.image("train/input image",
                         self.imageRearrange(tf.clip_by_value(self.images, 0, 1), block))
        tf.summary.image("train/gen image",
                         self.imageRearrange(tf.clip_by_value(self.G, 0, 1), block))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=2)


    def set_loss(self, G, G2, images, images2):
        im_id = tf.constant(np.random.choice(np.arange(self.batch_size), self.config.test_locations))
#         if 'optme' in self.config.model:
#             with tf.variable_scope('discriminator'):
#                 self.me_test_images = tf.get_variable(
#                     'd_me_test_images',
# #                        [self.batch_size, self.output_size, self.output_size, self.c_dim],
#                     initializer=self.additional_sample_images
#                 )
#             p = tf.cast(tf.reshape(tf.multinomial([[.5, .5]], self.batch_size),
#                                    [self.batch_size, 1, 1, 1]), tf.float32)
#             self.me_test_images = p * self.me_test_images + (1 - p) * self.sampler
#             meti = tf.clip_by_value(tf.gather(self.me_test_images, im_id), 0, 1)
#             bloc = int(np.floor(np.sqrt(self.config.test_locations)))
#             tf.summary.image("train/me test image", self.imageRearrange(meti, bloc))
#             if self.config.dc_discriminator:
#                 metl = self.discriminator(self.me_test_images, reuse=True)
#             else:
#                 metl = tf.reshape(self.me_test_images, [self.batch_size, -1])
#             self.me_test_locations = tf.gather(metl, im_id)
#         else:
        self.test_locations_ph = tf.placeholder(name='test_locations_ph', dtype=tf.float32,
                                                shape=(self.config.test_locations, self.dof_dim))
        self.test_locations = tf.concat([G2, images2], 0)
        self.test_locations_v = np.zeros((self.config.test_locations, self.dof_dim))


        assert self.config.kernel in ['dot', 'mix_rq', 'mix_rbf', 'distance'], \
            "Kernel '%s' not supported" % self.config.kernel
        kernel = getattr(MMD, '_%s_kernel' % self.config.kernel)

        self.mu_real = tf.reduce_mean(kernel(self.test_locations_ph, images, K_XY_only=True), axis=1)
        self.mu_real_ph = tf.placeholder(name='mu_real_ph', dtype=tf.float32,
                                         shape=(self.config.test_locations,))
        self.mu_real_value = np.zeros((self.config.test_locations,))
        mu_fake = tf.reduce_mean(kernel(self.test_locations_ph, G, K_XY_only=True), axis=1)
        diff = self.mu_real_ph - mu_fake

        self.optim_name = self.config.kernel + ' kernel mean embedding brb loss'
        with tf.variable_scope('loss'):
            self.g_loss = tf.sqrt(_eps + tf.reduce_sum(tf.square(diff))) / self.config.test_locations
#            Z = diff * self.batch_size
            self.d_loss = -tf.sqrt(_eps + tf.reduce_sum(tf.square(self.mu_real - mu_fake))) / self.config.test_locations

        self.d_loss_value, self.g_loss_value = np.inf, np.inf

        self.add_gradient_penalty(kernel, G, images)
        

    def add_gradient_penalty(self, kernel, fake, real):
        if self.config.gradient_penalty == 0:
            return
        bs = self.config.test_locations
        real, fake = real[:bs], fake[:bs]
        alpha = tf.random_uniform(shape=[bs])
        if 'mid' in self.config.suffix:
            alpha = .4 + .2 * alpha
        elif 'edges' in self.config.suffix:
            qq = tf.cast(tf.reshape(tf.multinomial([[.5, .5]], bs),
                                    [bs]), tf.float32)
            alpha = .1 * alpha * qq + (1. - .1 * alpha) * (1. - qq)
        elif 'edge' in self.config.suffix:
            alpha = .99 + .01 * alpha

        if self.config.gp_type == 'feature_space':
            alpha = tf.reshape(alpha, [bs, 1])
            x_hat = (1. - alpha) * real + alpha * fake
            Ekx = lambda yy: tf.reduce_mean(kernel(x_hat, yy, K_XY_only=True), axis=1)
            witness = Ekx(real) - Ekx(fake)
            gradients = tf.gradients(witness, [x_hat])[0]
        elif self.config.gp_type == 'data_space':
            alpha = tf.reshape(alpha, [bs, 1, 1, 1])
            real_data = self.images[:bs] #before discirminator
            fake_data = self.G[:bs] #before discriminator
            x_hat_data = (1. - alpha) * real_data + alpha * fake_data
            if self.check_numerics:
                x_hat_data = tf.check_numerics(x_hat_data, 'x_hat_data')
            x_hat = self.discriminator(x_hat_data, reuse=True, batch_size=bs)
            if self.check_numerics:
                x_hat = tf.check_numerics(x_hat, 'x_hat')
            Ekx = lambda yy: tf.reduce_mean(kernel(x_hat, yy, K_XY_only=True), axis=1)
            witness = Ekx(real) - Ekx(fake)
            if self.check_numerics:
                witness = tf.check_numerics(witness, 'witness')
            gradients = tf.gradients(witness, [x_hat_data])[0]
            if self.check_numerics:
                gradients = tf.check_numerics(gradients, 'gradients 0')
        elif self.config.gp_type == 'wgan':
            alpha = tf.reshape(alpha, [bs, 1, 1, 1])
            real_data = self.images #before discirminator
            fake_data = self.G #before discriminator
            x_hat_data = (1. - alpha) * real_data + alpha * fake_data
            x_hat = self.discriminator(x_hat_data, reuse=True, batch_size=bs)
            gradients = tf.gradients(x_hat, [x_hat_data])[0]
        
        if self.check_numerics:
#            gradients = tf.check_numerics(tf.clip_by_norm(gradients, 100.), 'gradients F')    
            penalty = tf.check_numerics(tf.reduce_mean(tf.square(tf.norm(gradients, axis=1) - 1.0)), 'penalty')
        else:
#            gradients = tf.clip_by_norm(gradients, 100.) 
            penalty = tf.reduce_mean(tf.square(tf.norm(gradients, axis=1) - 1.0))#

        
        print('adding gradient penalty')
        with tf.variable_scope('loss'):
            self.gp = tf.get_variable('gradient_penalty', dtype=tf.float32,
                                      initializer=self.config.gradient_penalty)
            self.d_loss += penalty * self.gp
            self.optim_name += ' gp %.1f' % self.config.gradient_penalty
            # variable_summaries([(gradients, 'dx_gradients')])
            tf.summary.scalar(self.optim_name + ' G', self.g_loss)
            tf.summary.scalar(self.optim_name + ' D', self.d_loss)
            tf.summary.scalar('dx_penalty', penalty)
        
            
    def train_step(self):
        step = self.sess.run(self.global_step)
        write_summary = ((np.mod(step, 50) == 0) and (step < 1000)) \
                    or (np.mod(step, 1000) == 0) or (self.err_counter > 0)
        # write_summary = True
        # print('d, g = %d, %d' % (self.d_counter, self.g_counter))
        feed_dict = {self.mu_real_ph: self.mu_real_value,
                     self.test_locations_ph: self.test_locations_v}
        if self.config.use_kernel:
            if self.config.is_demo:
                summary_str, step, self.g_loss_value, self.d_loss_value = self.sess.run(
                    [self.TrainSummary] + eval_ops
                )
            else:
                if self.g_counter == 0:
                    self.test_locations_v, self.mu_real_value, _, self.d_loss_value = self.sess.run(
                        [self.test_locations, self.mu_real, self.d_grads, self.d_loss],
                        feed_dict=feed_dict
                    )
                    assert ~np.isnan(self.d_loss_value), "NaN d_loss, epoch: [%2d] time: %4.4f" % (step, time.time() - self.start_time)
                else:
                    if write_summary:
                        _, summary_str, self.g_loss_value = self.sess.run(
                            [self.g_grads, self.TrainSummary, self.g_loss],
                            feed_dict=feed_dict
                        )
                    else:
                        _, self.g_loss_value = self.sess.run([self.g_grads, self.g_loss],
                                                             feed_dict=feed_dict)
                    assert ~np.isnan(self.g_loss_value), "NaN g_loss, epoch: [%2d] time: %4.4f" % (step, time.time() - self.start_time)

        if self.d_counter == 0:
            if write_summary:
                try:
                    self.writer.add_summary(summary_str, step)
                    self.err_counter = 0
                except Exception as e:
                    print('Step %d summary exception. ' % step, e)
                    self.err_counter += 1
                    
                print("Epoch: [%2d] time: %4.4f, %s, G: %.8f, D: %.8f"
                    % (step, time.time() - self.start_time, 
                       self.optim_name, self.g_loss_value, self.d_loss_value))
            if (np.mod(step + 1, self.config.max_iteration//5) == 0):
                self.lr *= self.config.decay_rate
                print('current learning rate: %f' % self.sess.run(self.lr))
                if ('decay_gp' in self.config.suffix) and (self.config.gradient_penalty > 0):
                    self.gp *= self.config.decay_rate
                    print('current gradient penalty: %f' % self.sess.run(self.gp))
                    
        if (step == 1) and (self.d_counter == 0):
            print('current learning rate: %f' % self.sess.run(self.lr))
        if (self.g_counter == 0) and (self.d_grads is not None):
            d_steps = self.config.dsteps
            if ((step % 100 == 0) or (step < 20)):
                d_steps = self.config.start_dsteps
            self.d_counter = (self.d_counter + 1) % (d_steps + 1)
        if self.d_counter == 0:
            self.g_counter = (self.g_counter + 1) % (self.config.gsteps + 1)
            self.d_counter += (self.g_counter == 0)
        return self.g_loss_value, self.d_loss_value, step
    