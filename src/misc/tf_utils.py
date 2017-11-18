import tensorflow as tf
import numpy as np


def fc(x, n_output, scope="fc", activation_fn=None, initializer=None):
    """fully connected layer with relu activation wrapper
    Args
      x:          2d tensor [batch, n_input]
      n_output    output size
    """
    with tf.variable_scope(scope):
        if initializer is None:
            # default initialization
            W = tf.Variable(tf.random_normal([int(x.get_shape()[1]), n_output]))
            b = tf.Variable(tf.random_normal([n_output]))
        else:
            W = tf.get_variable("W", shape=[int(x.get_shape()[1]), n_output], initializer=initializer)
            b = tf.get_variable("b", shape=[n_output],
                                initializer=tf.constant_initializer(.0, dtype=tf.float32))
        fc1 = tf.add(tf.matmul(x, W), b)
        if not activation_fn is None:
            fc1 = activation_fn(fc1)
    return fc1


def compile_function(inputs, outputs, log_name=None):
    def run(*input_vals):
        sess = tf.get_default_session()
        return sess.run(outputs, feed_dict=dict(zip(inputs, input_vals)))

    return run


def flatten_tensor_variables(ts):
    return tf.concat(0, [tf.reshape(x, [-1]) for x in ts])


def flatten_tensors(tensors):
    if len(tensors) > 0:
        return np.concatenate(map(lambda x: np.reshape(x, [-1]), tensors))
    else:
        return np.asarray([])


def unflatten_tensors(flattened, tensor_shapes):
    tensor_sizes = map(np.prod, tensor_shapes)
    indices = np.cumsum(tensor_sizes)[:-1]
    return map(lambda pair: np.reshape(pair[0], pair[1]), zip(np.split(flattened, indices), tensor_shapes))


def lrelu(x, leak=0.2):
    """
    Leaky ReLU
    """
    return tf.maximum(x, leak*x)