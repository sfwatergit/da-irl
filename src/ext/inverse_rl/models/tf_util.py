import tensorflow as tf
import numpy as np

REG_VARS = 'reg_vars'

def linear(X, dout, name, bias=True):
    with tf.variable_scope(name):
        dX = int(X.get_shape()[-1])
        W = tf.get_variable('W', shape=(dX, dout))
        tf.add_to_collection(REG_VARS, W)
        if bias:
            b = tf.get_variable('b', initializer=tf.constant(np.zeros(dout).astype(np.float32)))
        else:
            b = 0
    return tf.matmul(X, W)+b

def discounted_reduce_sum(X, discount, axis=-1):
    if discount != 1.0:
        disc = tf.cumprod(discount*tf.ones_like(X), axis=axis)
    else:
        disc = 1.0
    return tf.reduce_sum(X*disc, axis=axis)

def assert_shape(tens, shape):
    assert tens.get_shape().is_compatible_with(shape)

def relu_layer(X, dout, name):
    return tf.nn.relu(linear(X, dout, name))

def softplus_layer(X, dout, name):
    return tf.nn.softplus(linear(X, dout, name))

def tanh_layer(X, dout, name):
    return tf.nn.tanh(linear(X, dout, name))

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def minibatch(input, num_kernels=5, kernel_dim=3):
    x =  tf.contrib.layers.fully_connected(input, num_kernels*kernel_dim,
                                                     scope='minibatch')
    # is [?, ?, 1
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)