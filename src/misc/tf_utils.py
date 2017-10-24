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


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, int(1E+8)))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def lrelu(x, leak=0.2):
    """
    Leaky ReLU
    """
    return tf.maximum(x, leak*x)