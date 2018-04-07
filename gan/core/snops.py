from tensorflow.python.framework import ops
from utils.misc import variable_summaries
from .mmd import _eps, tf
from sn import spectral_normed_weight

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, scale = 1.0, is_train_scale = False, spectral_normed = False, 
           name="snconv2d",  update_collection = None):
    with tf.variable_scope(name):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                       tf.get_variable_scope().name)
        has_summary = any([('w' in v.op.name) for v in scope_vars])
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))


        if spectral_normed:
            s = tf.get_variable('s', shape =  [1],initializer = tf.constant_initializer(scale) ,trainable=is_train_scale,dtype=tf.float32)
            w_bar = s*spectral_normed_weight(w, update_collection=update_collection)
            conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        if not has_summary:
            if spectral_normed:
                variable_summaries({'W': w, 'b': biases, 's':s})
            else:
                variable_summaries({'W': w, 'b': biases})  
        
        return conv




def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,scale = 1.0, is_train_scale = False, spectral_normed = False, 
             name="deconv2d", with_w=False,  update_collection = None):
    with tf.variable_scope(name):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                       tf.get_variable_scope().name)
        has_summary = any([('w' in v.op.name) for v in scope_vars])
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        if spectral_normed:
            s = tf.get_variable('s', shape =  [1],initializer = tf.constant_initializer(scale) ,trainable=is_train_scale,dtype=tf.float32)
            w_bar = s*spectral_normed_weight(w, update_collection=update_collection)
            deconv = tf.nn.conv2d_transpose(input_, w_bar, output_shape=output_shape,strides=[1, d_h, d_w, 1])
        else:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        
        if not has_summary:
            if spectral_normed:
                variable_summaries({'W': w, 'b': biases, 's':s})
            else:
                variable_summaries({'W': w, 'b': biases})
            

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def linear(input_, output_size, name="Linear", stddev=0.01, scale = 1.0, is_train_scale = False, spectral_normed = False, bias_start=0.0, with_w=False, update_collection = None):
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(name):
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                       tf.get_variable_scope().name)
        has_summary = any([('Matrix' in v.op.name) for v in scope_vars])
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))

        if spectral_normed:
            s = tf.get_variable('s', shape =  [1],initializer = tf.constant_initializer(scale) ,trainable=is_train_scale,dtype=tf.float32)
            matrix_bar = s*spectral_normed_weight(matrix, update_collection=update_collection)
            mul = tf.matmul(input_, matrix_bar)
            
        else:
            mul = tf.matmul(input_, matrix)

        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        
        if not has_summary:
            if spectral_normed:
                variable_summaries({'W': matrix, 'b': bias, 's':s})
            else:
                variable_summaries({'W': matrix, 'b': bias})
        
        if with_w:
            return mul + bias, matrix, bias
        else:
            return mul + bias


