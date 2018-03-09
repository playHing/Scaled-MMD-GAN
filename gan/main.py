import sys
import numpy as np
import core
from utils.misc import pp, visualize

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-max_iteration',       default=400000, type=int, help='Epoch to train [400000]')
parser.add_argument('-learning_rate',       default=0.0001, type=float, help='Learning rate [2]')
parser.add_argument('-learning_rate_D',     default=-1, type=float, help='Learning rate for discriminator, if negative same as generator [-1]')
parser.add_argument('-MMD_lr_scheduler',    default=True, type=bool, help='Wheather to use lr scheduler based on 3-sample test')
parser.add_argument('-decay_rate',          default=.8, type=float, help='Decay rate [1.0]')
parser.add_argument('-gp_decay_rate',       default=.8, type=float, help='Decay rate [1.0]')
parser.add_argument('-beta1',               default=0.5, type=float, help='Momentum term of adam [0.5]')
parser.add_argument('-init',                default=0.02, type=float, help='Initialization value [0.02]')
parser.add_argument('-batch_size',          default=64, type=int, help='The size of batch images [1000]')
parser.add_argument('-real_batch_size',     default=-1 , type=int, help='The size of batch images for real samples. If -1 then same as batch_size [-1]')
parser.add_argument('-output_size',         default=128, type=int, help='The size of the output images to produce [64')
parser.add_argument('-c_dim',               default=3, type=int, help='Dimension of image color. [3]')
parser.add_argument('-dataset',             default="cifar10", type=str, help='The name of the model fro saving puposes')
parser.add_argument('-name',                default="mmd_test", type=str, help='The name of dataset [celebA, mnist, lsun, cifar10, GaussianMix]')
parser.add_argument('-checkpoint_dir',      default="checkpoint_mmd", type=str, help='Directory name to save the checkpoints [checkpoint_mmd]')
parser.add_argument('-sample_dir',          default="sample_mmd", type=str, help='Directory name to save the image samples [samples_mmd]')
parser.add_argument('-log_dir',             default="log_mmd", type=str, help='Directory name to save the image samples [logs_mmd]')
parser.add_argument('-data_dir',            default="data", type=str, help='Directory containing datasets [./data]')
parser.add_argument('-architecture',        default="dcgan", type=str, help='The name of the architecture [dcgan, g-resnet5, dcgan5]')
parser.add_argument('-kernel',              default="", type=str, help="The name of the architecture ['', 'mix_rbf', 'mix_rq', 'distance', 'dot', 'mix_rq_dot']")
parser.add_argument('-model',               default="mmd", type=str, help='The model type [mmd, cramer, wgan_gp]')
parser.add_argument('-is_train',            default=True, type=bool, help='True for training, False for testing [Train]')
parser.add_argument('-visualize',           default=False, type=bool, help='True for visualizing, False for nothing [False]')
parser.add_argument('-is_demo',             default=False, type=bool, help='For testing [False]')
parser.add_argument('-hessian_scale',       default=False, type=bool, help='For scaling the MMD')
parser.add_argument('-scale_variant',       default=0, type=int, help='The variant of the scaled MMD')
parser.add_argument('-gradient_penalty',    default=0.0, type=float, help='Use gradient penalty [0.0]')
parser.add_argument('-threads',             default=-1, type=int, help='Upper limit for number of threads [np.inf]')
parser.add_argument('-dsteps',              default=5, type=int, help='Number of discriminator steps in a row [1]')
parser.add_argument('-gsteps',              default=1, type=int, help='Number of generator steps in a row [1]')
parser.add_argument('-start_dsteps',        default=10, type=int, help='Number of discrimintor steps in a row during first 20 steps and every 100th step [1]')
parser.add_argument('-df_dim',              default=64, type=int, help='Discriminator no of channels at first conv layer [64]')
parser.add_argument('-dof_dim',             default=16, type=int, help='No of discriminator output features [16]')
parser.add_argument('-gf_dim',              default=61, type=int, help='No of generator channels [64]')
parser.add_argument('-batch_norm',          default=True, type=bool, help='Use of batch norm [False] (always False for discriminator if gradient_penalty > 0)')
parser.add_argument('-log',                 default=True, type=bool, help='Wheather to write log to a file in samples directory [True]')
parser.add_argument('-compute_scores',      default=False, type=bool, help='Compute scores')
parser.add_argument('-print_pca',           default=False, type=bool, help='Print the PCA')
parser.add_argument('-suffix',              default="", type=str, help="Fo additional settings ['', '_tf_records']")
parser.add_argument('-gpu_mem',             default=.9, type=float, help="GPU memory fraction limit [1.0]")
parser.add_argument('-no_of_samples',       default=100000, type=int, help="number of samples to produce")
parser.add_argument('-save_layer_outputs',  default=0, type=int, help="Wheather to save_layer_outputs. If == 2, saves outputs at exponential steps: 1, 2, 4, ..., 512 and every 1000. [0, 1, 2]")
parser.add_argument('-L2_discriminator_penalty',default=0.0, type=float, help="L2 penalty on discriminator features [0.0]")
FLAGS = parser.parse_args()   

def main(_):
    pp.pprint(vars(FLAGS))
        
    if FLAGS.threads > -1:
        sess_config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.threads)
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_mem
        
    else:
        sess_config = tf.ConfigProto()
    if FLAGS.model == 'mmd':
        from core.model import MMD_GAN as Model
    elif FLAGS.model == 'wgan_gp':
        from core.wgan_gp import WGAN_GP as Model
    elif 'cramer' in FLAGS.model:
        from core.cramer import Cramer_GAN as Model
    elif FLAGS.model == 'wmmd':
        from core.wmmd import WMMD as Model

    with tf.Session(config=sess_config) as sess:
        #sess = tf_debug.tf_debug.TensorBoardDebugWrapperSession(sess,'localhost:6064')
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
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
