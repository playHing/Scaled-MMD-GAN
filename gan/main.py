import sys
import numpy as np
import core
from utils.misc import pp, visualize

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
flags.DEFINE_integer("batch_size", 128, "The size of batch images [1000]")
flags.DEFINE_integer("real_batch_size", -1, "The size of batch images for real samples. If -1 then same as batch_size [-1]")
flags.DEFINE_integer("output_size", 32, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "cifar10", "The name of the model fro saving puposes")
flags.DEFINE_string("name", "mmd_test", "The name of dataset [celebA, mnist, lsun, cifar10, GaussianMix]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_mmd", "Directory name to save the checkpoints [checkpoint_mmd]")
flags.DEFINE_string("sample_dir", "samples_mmd", "Directory name to save the image samples [samples_mmd]")
flags.DEFINE_string("log_dir", "logs_mmd", "Directory name to save the image samples [logs_mmd]")
flags.DEFINE_string("data_dir", "./data", "Directory containing datasets [./data]")
flags.DEFINE_string("architecture", "dcgan", "The name of the architecture [dcgan, g-resnet5, dcgan5]")
flags.DEFINE_string("kernel", "", "The name of the architecture ['', 'mix_rbf', 'mix_rq', 'distance', 'dot', 'mix_rq_dot']")
flags.DEFINE_string("model", "mmd", "The model type [mmd, cramer, wgan_gp]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [Train]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("is_demo", False, "For testing [False]")
flags.DEFINE_float("gradient_penalty", 0.0, "Use gradient penalty [0.0]")
flags.DEFINE_integer("threads", np.inf, "Upper limit for number of threads [np.inf]")
flags.DEFINE_integer("dsteps", 1, "Number of discriminator steps in a row [1] ")
flags.DEFINE_integer("gsteps", 1, "Number of generator steps in a row [1] ")
flags.DEFINE_integer("start_dsteps", 1, "Number of discrimintor steps in a row during first 20 steps and every 100th step" [1])
flags.DEFINE_integer("df_dim", 64, "Discriminator no of channels at first conv layer [64]")
flags.DEFINE_integer("dof_dim", 16, "No of discriminator output features [16]")
flags.DEFINE_integer("gf_dim", 64, "no of generator channels [64]")
flags.DEFINE_boolean("batch_norm", True, "Use of batch norm [False] (always False for discriminator if gradient_penalty > 0)")
flags.DEFINE_boolean("log", True, "Wheather to write log to a file in samples directory [True]")
flags.DEFINE_string("suffix", '', "Fo additional settings ['', '_tf_records']")
flags.DEFINE_boolean('compute_scores', False, "Compute scores")
flags.DEFINE_float("gpu_mem", .9, "GPU memory fraction limit [1.0]")
flags.DEFINE_float("L2_discriminator_penalty", 0.0, "L2 penalty on discriminator features [0.0]")
flags.DEFINE_integer("no_of_samples", 100000, "number of samples to produce")
flags.DEFINE_boolean("print_pca", False, "")
flags.DEFINE_integer("save_layer_outputs", 0, "Wheather to save_layer_outputs. If == 2, saves outputs at exponential steps: 1, 2, 4, ..., 512 and every 1000. [0, 1, 2]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(FLAGS.__flags)
        
    if FLAGS.threads < np.inf:
        sess_config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.threads)
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_mem
        
    else:
        sess_config = tf.ConfigProto()
    if 'mmd' in FLAGS.model:
        from core.model import MMD_GAN as Model
    elif FLAGS.model == 'wgan_gp':
        from core.wgan_gp import WGAN_GP as Model
    elif 'cramer' in FLAGS.model:
        from core.cramer import Cramer_GAN as Model

        
    with tf.Session(config=sess_config) as sess:
        if FLAGS.dataset == 'mnist':
            gan = Model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=28, c_dim=1,
                        data_dir=FLAGS.data_dir)
        elif FLAGS.dataset == 'cifar10':
            gan = Model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=32, c_dim=3,
                        data_dir=FLAGS.data_dir)
        elif FLAGS.dataset in  ['celebA', 'lsun']:
            gan = Model(sess, config=FLAGS, batch_size=FLAGS.batch_size, output_size=FLAGS.output_size, c_dim=3,
                        data_dir=FLAGS.data_dir)
        else:
            gan = Model(sess, batch_size=FLAGS.batch_size, 
                        output_size=FLAGS.output_size, c_dim=FLAGS.c_dim,
                        data_dir=FLAGS.data_dir)
            
        if FLAGS.is_train:
            gan.train()
        elif FLAGS.print_pca:
            gan.print_pca()
        elif FLAGS.visualize:
            gan.load_checkpoint()
            visualize(sess, gan, FLAGS, 2)
        else:
            gan.get_samples(FLAGS.no_of_samples, layers=[-1])


        if FLAGS.log:
            sys.stdout = gan.old_stdout
            gan.log_file.close()
        gan.sess.close()
        
if __name__ == '__main__':
    tf.app.run()
