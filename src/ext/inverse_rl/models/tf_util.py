import re

import collections
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl

REG_VARS = 'reg_vars'


def linear(X, dout, name, bias=True):
    with tf.variable_scope(name):
        dX = int(X.get_shape()[-1])
        W = tf.get_variable('W', shape=(dX, dout))
        tf.add_to_collection(REG_VARS, W)
        if bias:
            b = tf.get_variable('b', initializer=tf.constant(
                np.zeros(dout).astype(np.float32)))
        else:
            b = 0
    return tf.matmul(X, W) + b


def discounted_reduce_sum(X, discount, axis=-1):
    if discount != 1.0:
        disc = tf.cumprod(discount * tf.ones_like(X), axis=axis)
    else:
        disc = 1.0
    return tf.reduce_sum(X * disc, axis=axis)


def assert_shape(tens, shape):
    assert tens.get_shape().is_compatible_with(shape)


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(tcl.batch_norm(x), alpha)


def relu_layer(X, dout, name):
    return tf.nn.swish(linear(X, dout, name))


def softplus_layer(X, dout, name):
    return tf.nn.softplus(linear(X, dout, name))


def tanh_layer(X, dout, name):
    return tf.nn.tanh(linear(X, dout, name))


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def gradient_summaries(grad_vars, groups=None, scope='gradients'):
    """Create histogram summaries of the gradient.
    Summaries can be grouped via regexes matching variables names.
    Args:
      grad_vars: List of (gradient, variable) tuples as returned by optimizers.
      groups: Mapping of name to regex for grouping summaries.
      scope: Name scope for this operation.
    Returns:
      Summary tensor.
    """
    groups = groups or {r'all': r'.*'}
    grouped = collections.defaultdict(list)
    for grad, var in grad_vars:
        if grad is None:
            continue
        for name, pattern in groups.items():
            if re.match(pattern, var.name):
                name = re.sub(pattern, name, var.name)
                grouped[name].append(grad)
    for name in groups:
        if name not in grouped:
            tf.logging.warn("No variables matching '{}' group.".format(name))
    summaries = []
    for name, grads in grouped.items():
        grads = [tf.reshape(grad, [-1]) for grad in grads]
        grads = tf.concat(grads, 0)
        summaries.append(tf.summary.histogram(scope + '/' + name, grads))
    return tf.summary.merge(summaries)


def variable_summaries(vars_, groups=None, scope='weights'):
    """Create histogram summaries for the provided variables.
    Summaries can be grouped via regexes matching variables names.
    Args:
      vars_: List of variables to summarize.
      groups: Mapping of name to regex for grouping summaries.
      scope: Name scope for this operation.
    Returns:
      Summary tensor.
    """
    groups = groups or {r'all': r'.*'}
    grouped = collections.defaultdict(list)
    for var in vars_:
        for name, pattern in groups.items():
            if re.match(pattern, var.name):
                name = re.sub(pattern, name, var.name)
                grouped[name].append(var)
    for name in groups:
        if name not in grouped:
            tf.logging.warn("No variables matching '{}' group.".format(name))
    summaries = []
    # pylint: disable=redefined-argument-from-local
    for name, vars_ in grouped.items():
        vars_ = [tf.reshape(var, [-1]) for var in vars_]
        vars_ = tf.concat(vars_, 0)
        summaries.append(tf.summary.histogram(scope + '/' + name, vars_))
    return tf.summary.merge(summaries)
