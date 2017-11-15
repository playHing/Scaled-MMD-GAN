import os, sys
import scipy.misc
import numpy as np

from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("max_iteration", 400000, "Epoch to train [400000]")
flags.DEFINE_float("learning_rate", 2, "Learning rate [2]")
flags.DEFINE_float("learning_rate_D", -1, "Learning rate for discriminator, if negative same as generator [-1]")
flags.DEFINE_boolean("MMD_lr_scheduler", True, "Wheather to use lr scheduler based on 3-sample test")
flags.DEFINE_float("decay_rate", .5, "Decay rate [1.0]")
flags.DEFINE_float("gp_decay_rate", .5, "Decay rate [1.0]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("init", 0.02, "Initialization value [0.02]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [1000]")
flags.DEFINE_integer("real_batch_size", -1, "The size of batch images for real samples [1000]")
flags.DEFINE_integer("output_size", 32, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "cifar10", "The name of dataset [celebA, mnist, lsun, cifar10, GaussianMix]")
flags.DEFINE_string("name", "mmd_test", "The name of dataset [celebA, mnist, lsun, cifar10, GaussianMix]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_mmd", "Directory name to save the checkpoints [checkpoint_mmd]")
flags.DEFINE_string("sample_dir", "samples_mmd", "Directory name to save the image samples [samples_mmd]")
flags.DEFINE_string("log_dir", "logs_mmd", "Directory name to save the image samples [logs_mmd]")
flags.DEFINE_string("data_dir", "./data", "Directory containing datasets [./data]")
flags.DEFINE_string("architecture", "dc", "The name of the architecture [dc, mlp, dfc]")
flags.DEFINE_string("kernel", "", "The name of the architecture ['', 'mix_rbf', 'mix_rq', 'distance', 'dot']")
flags.DEFINE_string("model", "mmd", "The name of the kernel loss model [mmd, tmmd, me]")
flags.DEFINE_boolean("dc_discriminator", False, "use deep convolutional discriminator [True]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("is_demo", False, "For testing [False]")
flags.DEFINE_float("gradient_penalty", 0.0, "Use gradient penalty [0.0]")
flags.DEFINE_float("discriminator_weight_clip", 0.0, "Use discriminator weight clip [0.0]")
flags.DEFINE_integer("threads", np.inf, "Upper limit for number of threads [np.inf]")
flags.DEFINE_integer("dsteps", 1, "Number of discriminator steps in a row [1] ")
flags.DEFINE_integer("gsteps", 1, "Number of generator steps in a row [1] ")
flags.DEFINE_integer("start_dsteps", 1, "Number of discrimintor steps in a row during first 20 steps and every 100th step" [1])
flags.DEFINE_integer("df_dim", 64, "Discriminator no of channels at first conv layer [64]")
flags.DEFINE_integer("dof_dim", 16, "No of discriminator output features [16]")
flags.DEFINE_integer("gf_dim", 64, "no of generator channels [64]")
flags.DEFINE_boolean("batch_norm", False, "Use of batch norm [False] (always False for discriminator if gradient_penalty > 0)")
flags.DEFINE_integer("test_locations", 16, "No of test locations for mean-embedding model [16] ")
flags.DEFINE_boolean("log", True, "Wheather to write log to a file in samples directory [True]")
flags.DEFINE_string("suffix", '', "Additional settings ['']")
flags.DEFINE_string("gp_type", 'data_space', "type of gradient penalty ['data_space', 'feature_space', 'wgan']")
flags.DEFINE_string("single_batch_experiment", False, "wheater to train on a constant single batch with constant gp evaluation points")
flags.DEFINE_boolean('compute_scores', False, "Compute scores")
flags.DEFINE_float("gpu_mem", .9, "GPU memory fraction limit [1.0]")
flags.DEFINE_float("L2_discriminator_penalty", 0.0, "L2 penalty on discriminator features [0.0]")
FLAGS = flags.FLAGS

def main(_):
    if FLAGS.architecture not in ['dfc', 'dcold']:
        if FLAGS.dof_dim > 0:
            FLAGS.architecture += '.' + str(FLAGS.dof_dim)

    pp.pprint(FLAGS.__flags)
        
    if FLAGS.threads < np.inf:
        sess_config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.threads)
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_mem
        
    else:
        sess_config = tf.ConfigProto()
    if FLAGS.model in ['mmd_gan']:
        from model_mmd_gan import MMDCE_GAN as model
    elif FLAGS.model in ['tmmd', 'mmd']:
        from model_mmd2 import MMD_GAN as model
    elif (FLAGS.model == 'me') or ('optme' in FLAGS.model):
        from model_me2 import ME_GAN as model
    elif FLAGS.model == 'me_brb':
        from model_me_brb import MEbrb_GAN as model
    elif FLAGS.model == 'gan':
        from model_gan import GAN as model
    elif FLAGS.model == 'wgan_gp':
        from model_wgan_gp import GAN as model
    elif 'cramer' in FLAGS.model:
        from cramer import Cramer_GAN as model

        
    with tf.Session(config=sess_config) as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=28, c_dim=1,
                          data_dir=FLAGS.data_dir)
        elif FLAGS.dataset == 'cifar10':
            dcgan = model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=32, c_dim=3,
                          data_dir=FLAGS.data_dir)
        elif ('lsun' in FLAGS.dataset) or (FLAGS.dataset == 'celebA'):
            dcgan = model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=FLAGS.output_size, c_dim=3,
                          data_dir=FLAGS.data_dir)
        elif FLAGS.dataset == 'GaussianMix':
            dcgan = model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=1, c_dim=1, z_dim=5,
                          data_dir=FLAGS.data_dir)
        else:
            dcgan = model(sess, batch_size=FLAGS.batch_size, 
                          output_size=FLAGS.output_size, c_dim=FLAGS.c_dim,
                          data_dir=FLAGS.data_dir)
            
        if FLAGS.is_train:
            dcgan.train()
        else:
            dcgan.get_samples(100000)

        if FLAGS.visualize:
            to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
                                          [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
                                          [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
                                          [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
                                          [dcgan.h4_w, dcgan.h4_b, None])

            # Below is codes for visualization
            OPTION = 2
            visualize(sess, dcgan, FLAGS, OPTION)

        if FLAGS.log:
            sys.stdout = dcgan.old_stdout
            dcgan.log_file.close()
        dcgan.sess.close()
        
if __name__ == '__main__':
    tf.app.run()
