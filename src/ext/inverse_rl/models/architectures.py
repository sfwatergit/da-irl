import tensorflow as tf
from ext.inverse_rl.models.tf_util import relu_layer, linear
from impl.timed_activities.timed_activity_rewards import TimedActivityRewards


def make_relu_net(layers=2, dout=1, d_hidden=128):
    def relu_net(x, last_layer_bias=True):
        out = x
        for i in range(layers):
            out = relu_layer(out, dout=d_hidden, name='l%d'%i)
        out = linear(out, dout=dout, name='lfinal', bias=last_layer_bias)
        return out
    return relu_net


def relu_net(x, layers=2, dout=1, d_hidden=256):
    out = x
    for i in range(layers):
        out = relu_layer(out, dout=d_hidden, name='l%d'%i)
    out = linear(out, dout=dout, name='lfinal')
    return out


def linear_net(x, dout=1):
    out = x
    out = linear(out, dout=dout, name='lfinal')
    return out


def feedforward_energy(obs_act, ff_arch=relu_net, dropout=True):
    # for trajectories, using feedforward nets rather than RNNs
    dimOU = int(obs_act.get_shape()[2])
    orig_shape = tf.shape(obs_act)

    obs_act = tf.reshape(obs_act, [-1, dimOU], name='ff_reshape_in')
    outputs = ff_arch(obs_act)
    dOut = int(outputs.get_shape()[-1])

    new_shape = tf.stack([orig_shape[0],orig_shape[1], dOut],
                         name="ff_stack_out")
    outputs = tf.reshape(outputs, new_shape, name="ff_reshape_out")
    return outputs


def rnn_trajectory_energy(obs_act):
    """
    Operates on trajectories
    """
    # for trajectories
    dimOU = int(obs_act.get_shape()[2])
    cell = tf.contrib.rnn.LSTMCell(num_units=128)
    cell_out = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
    outputs, hidden = tf.nn.dynamic_rnn(cell_out, obs_act, time_major=False, dtype=tf.float32)
    return outputs
