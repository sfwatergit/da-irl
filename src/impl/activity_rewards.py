import os
import platform

import matplotlib
import numpy as np
import tensorflow as tf

from src.core.mdp import RewardFunction
from src.impl.activity_env import ActivityEnv
from src.impl.activity_features import ActivityFeature, create_act_at_x_features, TripFeature
from src.util.math_utils import get_subclass_list, cartesian, create_dir_if_not_exists, normalize
from src.util.tf_utils import fc_net

if platform.system() == 'Darwin':
    matplotlib.rcParams['backend'] = 'agg'
else:
    matplotlib.rcParams['backend'] = 'Agg'

import matplotlib.pyplot as plt

plt.interactive(False)


class ActivityRewardFunction(RewardFunction):
    """
    Computes the activity reward based on the state which is the current activity and time of day.
    Initialized with config-defined scoring parameters.
    """

    def __init__(self, env, opt_params=None, nn_params=None, rmax=1.0,
                 initial_theta=None):
        # type: (ActivityEnv) -> None
        params = env.config
        self.activity_features = self.make_activity_features(env, params)
        self.trip_features = self.make_trip_features(env, params)
        self._make_indices(params)
        super(ActivityRewardFunction, self).__init__(self.activity_features + self.trip_features, env, rmax,
                                                     initial_weights=initial_theta)

        if nn_params is None:
            nn_params = {'h_dim': 32, 'reg_dim': 10, 'name': 'maxent_irl'}

        if opt_params is None:
            opt_params = {'lr': 0.3}

        self.lr = opt_params['lr']

        self.name = nn_params['name']
        self.h_dim = nn_params['h_dim']
        self.reg_dim = nn_params['reg_dim']

        self.input_size = self.dim_ss

        self.input_ph = tf.placeholder(tf.float32, shape=[None, self.input_size], name='dim_ss')

        with tf.variable_scope(self.name) as va:
            reward = fc_net(self.input_ph, n_layers=1, dim_hidden=self.h_dim, out_act=None,
                            init=initial_theta, name=self.name)
        self.theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        self.reward = reward

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        self.grad_r = tf.placeholder(tf.float32, [None, 1])

        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
        self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

        self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)

        self.grad_theta = [tf.add(self.reg_dim * self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
        # self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 10.0)

        self.grad_norms = tf.global_norm(self.grad_theta)
        self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def make_trip_features(env, params):
        return [i(mode, params, env=env) for i in get_subclass_list(TripFeature) for mode in
                params.travel_params.keys()]

    @staticmethod
    def make_activity_features(env, params):
        activity_features = [i(params, env=env) for i in get_subclass_list(ActivityFeature)]
        acts = [env.home_activity, env.work_activity, env.other_activity]
        time_range = np.arange(0, env.horizon * env.segment_minutes, env.segment_minutes)
        prod = cartesian([acts, time_range])
        act_at_x_features = [create_act_at_x_features(where, when, env.segment_minutes, params)(env=env)
                             for where, when in prod]
        activity_features += act_at_x_features
        return activity_features

    def _make_indices(self, params):
        assert (len(self.activity_features) > 0) and (len(self.trip_features) > 0)
        self._activity_feature_ixs = range(len(self.activity_features))
        self._trip_feature_ixs = range(len(self.activity_features), len(self.activity_features) +
                                       len(params.travel_params.keys()))

    def phi(self, s, a):
        phi = np.zeros((self._dim_ss, 1), float)
        state = self._env.states[s]
        feature_ixs = range(self._dim_ss)
        action = self._env.actions[a]
        for ix in feature_ixs:
            phi[ix] = self.features[ix](state, action)
        return phi

    def apply_grads(self, grad_r):
        feat_map = self.feature_matrix
        grad_r = np.reshape(grad_r, [-1, 1])
        feat_map = np.reshape(feat_map, [-1, self.dim_ss])
        _, grad_theta, l2_loss, grad_norms = self.sess.run(
            [self.optimize, self.grad_theta, self.l2_loss, self.grad_norms],
            feed_dict={self.grad_r: grad_r, self.input_ph: feat_map})
        return grad_theta, l2_loss, grad_norms

    def get_theta(self):
        return normalize(self.sess.run(self.theta)[0])

    def get_rewards(self):
        feed_dict = {self.input_ph: self.feature_matrix.reshape([-1, self.dim_ss])}
        rewards = self.sess.run(self.reward, feed_dict)
        return rewards.reshape([self._env.nS, self._env.nA])

    def plot_current_theta(self, ident):
        image_path = os.path.join(self._env.config.general_params.log_dir, os.path.join("expert_{}".format(ident)),self._env.config.general_params.images_dir)
        create_dir_if_not_exists(image_path)
        home_int = [self.activity_features.index(feat) for feat in self.activity_features if feat.ident.startswith('H')]
        work_int = [self.activity_features.index(feat) for feat in self.activity_features if feat.ident.startswith('W')]
        other_int = [self.activity_features.index(feat) for feat in self.activity_features if feat.ident.startswith('o')]
        plot_theta(self.get_theta(),home_int,work_int,other_int, self._env.segment_minutes, image_path, ident=ident)


def plot_theta(theta, home_int, work_int, other_int, disc_len,  image_path, show=False, ident=''):
    home_feats = theta[home_int[:-1]]  # last home activity is anchor (creates negative utility)
    work_feats = theta[work_int]
    other_feats = theta[other_int]
    plot_reward(home_feats, disc_len, image_path, 'home', 'b', show, ident)
    plt.clf()
    plot_reward(work_feats, disc_len, image_path, 'work', 'g', show, ident)
    plt.clf()
    plot_reward(other_feats, disc_len, image_path, 'other', 'r', show, ident)
    plt.clf()


def plot_reward(ys, disc_len=15., image_path='', title='', color='b', show=False, ident=''):
    xs = np.arange(0, len(ys)) * float(disc_len) / 60
    plt.plot(xs, ys, color)
    plt.title('Marginal Utility vs. Time of Day for {} Activity'.format(title.capitalize()))
    plt.xlabel('time (hr)')
    plt.ylabel('marginal utility (utils/hr)')
    if show:
        plt.show()
    else:
        plt.savefig(image_path + '/{}_activity_persona_{}'.format(title, ident))
