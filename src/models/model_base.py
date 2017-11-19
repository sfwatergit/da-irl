import tensorflow as tf

from misc.misc_utils import lazy_property


class Model:
    def __init__(self, nn_params):
        self.x = tf.placeholder(tf.float32, shape=[None, 3])
        self.y_true = tf.placeholder(tf.float32, shape=None)
        self.params = self._initialize_weights()
        self.lr = nn_params['lr']
        self.n_iter = nn_params['n_iter']

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self._output = None
        self._optimizer = None
        self._loss = None

    @staticmethod
    def _initialize_weights():
        params = dict()
        params['w'] = tf.Variable([0, 0, 0], dtype=tf.float32)
        params['b'] = tf.Variable(0, dtype=tf.float32)
        return params

    @lazy_property
    def output(self):
        return tf.matmul(self.params['w'], tf.transpose(self.x)) + self.params['b']

    @lazy_property
    def loss(self):
        return tf.reduce_mean(tf.square(self.y_true - self.output))

    @lazy_property
    def optimizer(self):
        return tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def fit(self, x_train, y_train):
        for itr in range(self.n_iter):
            self.sess.run(self.optimizer, {self.x: x_train, self.y_true: y_train})

    def evaluate(self, x_test, y_test):
        return self.sess.run(self.loss, {self.x: x_test, self.y_true: y_test})