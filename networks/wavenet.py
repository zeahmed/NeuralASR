import numpy as np
import tensorflow as tf

from .common import (label_error_rate, loss, model, setup_training_network,
                     variable_on_worker_level)

num_blocks = 3     # dilated blocks
num_dim = 128      # latent dimension


def _get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        kernel_size = np.prod(shape[:2])
        fan_in = shape[-2] * kernel_size
        fan_out = shape[-1] * kernel_size
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def uniform(name, shape, scale=0.05, dtype=tf.float32, summary=False, regularizer=None, trainable=True):
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    x = tf.get_variable(name, shape, dtype=dtype,
                        initializer=tf.random_uniform_initializer(
                            minval=-scale, maxval=scale),
                        regularizer=regularizer, trainable=trainable)
    # add summary
    if summary:
        tf.sg_summary_param(x)
    return x


def he_uniform(name, shape, scale=1, dtype=tf.float32, summary=False, regularizer=None, trainable=True):
    fin, _ = _get_fans(shape)
    s = np.sqrt(1. * scale / fin)
    return uniform(name, shape, s, dtype, summary, regularizer, trainable)


def constant(name, shape, value=0, dtype=tf.float32, summary=False, regularizer=None, trainable=True):
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    x = tf.get_variable(name, shape, dtype=dtype,
                        initializer=tf.constant_initializer(value),
                        regularizer=regularizer, trainable=trainable)
    # add summary
    if summary:
        tf.sg_summary_param(x)
    return x


def conv1d(tensor, size, dim, stride=1, pad='SAME', act=None, bn=False, bias=False, name=None, is_training=None):

    with tf.variable_scope(name):

        shape = tensor.get_shape().as_list()
        in_dim = shape[-1]

        # parameter tf.sg_initializer
        w = he_uniform('W', (size, in_dim, dim))
        b = constant('b', dim) if bias else 0

        out = tf.nn.conv1d(tensor, w, stride=stride, padding=pad) + b

        if bn:
            #out = batch_norm(out, dim, is_training)
            #out = batch_norm(out, name, is_training)
            out = tf.expand_dims(out, axis=2)
            out = tf.contrib.layers.batch_norm(out,
                                               decay=0.99,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               data_format='NHWC',
                                               zero_debias_moving_mean=True,
                                               is_training=is_training)
            out = tf.squeeze(out, axis=2)

        if act:
            out = act(out)

    return out


def dilated_conv1d(tensor, size, rate, dim, pad='SAME', act=None, bn=False, bias=False, name=None, is_training=None):

    with tf.variable_scope(name):

        shape = tensor.get_shape().as_list()
        in_dim = shape[-1]

        # parameter tf.sg_initializer
        w = he_uniform('W', (1, size, in_dim, dim))
        b = constant('b', dim) if bias else 0

        # apply 2d convolution
        out = tf.nn.atrous_conv2d(tf.expand_dims(tensor, axis=1),
                                  w, padding=pad, rate=rate) + b

        if bn:
            #out = batch_norm(out, dim, is_training)
            #out = batch_norm(out, name, is_training)
            out = tf.expand_dims(out, axis=2)
            out = tf.contrib.layers.batch_norm(out,
                                               decay=0.99,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               data_format='NHWC',
                                               zero_debias_moving_mean=True,
                                               is_training=is_training)
            out = tf.squeeze(out, axis=2)

        if act:
            out = act(out)
        # reduce dimension
        # noinspection PyUnresolvedReferences
        out = tf.squeeze(out, axis=1)

    return out


def create_network(features, seq_len, num_classes, is_training):
    '''
    This network is similar to wavenet https://github.com/buriburisuri/speech-to-text-wavenet
    '''
    # residual block
    def res_block(tensor, size, rate, block, dim=num_dim):

        name = 'block_%d_%d' % (block, rate)
        with tf.variable_scope(name_or_scope=name):

            # filter convolution
            conv_filter = dilated_conv1d(
                tensor, size=size, rate=rate, dim=dim, act=tf.nn.tanh, bn=True, name='conv_filter' + name, is_training=is_training)

            # gate convolution
            conv_gate = dilated_conv1d(
                tensor, size=size, rate=rate, dim=dim, act=tf.nn.sigmoid, bn=True, name='conv_gate' + name, is_training=is_training)

            # output by gate multiplying
            out = conv_filter * conv_gate

            # final output
            out = conv1d(out, size=1, dim=dim, act=tf.nn.tanh, bn=True,
                         name='conv_out' + name, is_training=is_training)

            # residual and skip output
            return out + tensor, out

    # expand dimension
    with tf.variable_scope(name_or_scope='front'):
        z = conv1d(features, size=1, dim=num_dim,
                   act=tf.nn.tanh, bn=True, name='conv_in', is_training=is_training)

    # dilated conv block loop
    skip = 0  # skip connections
    for i in range(num_blocks):
        for r in [1, 2, 4, 8, 16]:
            z, s = res_block(z, size=7, rate=r, block=i)
            skip += s

    # final logit layers
    with tf.variable_scope(name_or_scope='logit'):
        skip2 = conv1d(skip, size=1, dim=num_dim,
                       act=tf.nn.tanh, bn=True, name='conv_1', is_training=is_training)
        logits = conv1d(skip2, size=1, dim=num_classes,
                        name='conv_2', is_training=is_training)

    logits = tf.transpose(logits, (1, 0, 2))
    return logits
