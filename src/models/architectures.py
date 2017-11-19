import tensorflow as tf

from misc.tf_utils import fc


def make_fc_net(X, n_layers=2, dim_out=1, dim_hidden=32, act=tf.nn.elu,
                init=tf.contrib.layers.variance_scaling_initializer(mode=u'FAN_IN')):
    out = X
    for i in range(n_layers):
        out = fc(out, dim_hidden, scope=u'fc%d' % i, activation_fn=act,
                 initializer=init)
    out = fc(out, dim_out, scope=u'fc%s' % u'out', activation_fn=act,
             initializer=init)
    return out
