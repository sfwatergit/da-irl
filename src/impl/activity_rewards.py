import numpy as np
import tensorflow as tf
from cytoolz import memoize

from impl.activity_env import ActivityEnv
from models.architectures import make_fc_net
from src.core.mdp import RewardFunction
from src.impl.activity_features import ActivityFeature, create_act_at_x_features, TripFeature
from src.misc.math_utils import get_subclass_list, cartesian


class ActivityRewardFunction(RewardFunction):
    """
    Computes the activity reward based on the state which is the current activity and time of day.
    Initialized with config-defined scoring parameters.
    """

    def __init__(self, env):
        # type: (ActivityEnv) -> None
        params = env.params

        self.activity_features = self.make_activity_features(env, params)
        self.trip_features = self.make_trip_features(env, params)
        self._make_indices(params)

        super(ActivityRewardFunction, self).__init__(self.activity_features + self.trip_features, env)

    @staticmethod
    def make_trip_features(env, params):
        return [i(mode, params, env=env) for i in get_subclass_list(TripFeature) for mode in
                params.travel_params.keys()]

    @staticmethod
    def make_activity_features(env, params):
        activity_features = [i(params, env=env) for i in get_subclass_list(ActivityFeature)]
        acts = [env.home_act, env.work_act, env.shopping_act]
        time_range = np.arange(0, env.horizon * env.segment_mins, env.segment_mins)
        prod = cartesian([acts, time_range])
        act_at_x_features = [create_act_at_x_features(where, when, env.segment_mins, params)(env=env)
                             for where, when in prod]
        activity_features += act_at_x_features
        return activity_features

    def _make_indices(self, params):
        assert (len(self.activity_features) > 0) and (len(self.trip_features) > 0)
        self._activity_feature_ixs = range(len(self.activity_features))
        self._trip_feature_ixs = range(len(self.activity_features), len(self.activity_features) +
                                       len(params.travel_params.keys()))

    @memoize
    def __call__(self, state, action):
        """
        Args:
            state: denotes
            action: s'(t+1) denotes the decision of next state

        Returns:

        """
        phi = self.phi(state, action)
        return np.dot(self._weights, phi)

    def update_weights(self, weights):
        """ Update the weights parameters of the reward function model """
        assert weights.shape == self._weights.shape, \
            'New weight array size must match reward function dimension'
        self._weights = weights

    def update_reward(self, theta):
        r = np.zeros((self._env.nS, self._env.nA))
        for state in self._env.states.values():
            for a in state.available_actions:
                s = state.state_id
                r[s, a] = np.dot(theta, self.phi(s, a))
        self._r = r

    def get_reward(self):
        return self._r

    @memoize
    def phi(self, s, a):
        phi = np.zeros((self._dim_ss, 1), float)
        state = self._env.states[s]
        feature_ixs = range(self._dim_ss)
        if s in self._env.terminals or a == -1:
            return phi
        else:
            action = self._env.actions[a]
        for ix in feature_ixs:
            phi[ix] = self.features[ix](state, action)
        return phi


class ActivityNNReward(ActivityRewardFunction):
    def __init__(self, env=None, opt_params=None, nn_params=None):
        super(ActivityNNReward, self).__init__(env)

        if nn_params is None:
            nn_params = {'h_dim': 32, 'reg_dim': 10, 'name': 'maxent_deep_irl'}

        if opt_params is None:
            opt_params = {'lr': 0.01}

        self.lr = opt_params['lr']

        self.sess = tf.Session()

        self.name = nn_params['name']
        self.h_dim = nn_params['h_dim']
        self.reg_dim = nn_params['reg_dim']

        input_size = self.dim_ss

        self.input_ph = tf.placeholder(tf.float32, [None, input_size])

        with tf.variable_scope(self.name):
            reward = make_fc_net(self.input_ph, self.h_dim)
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        self.reward = reward
        self.theta = theta

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        self.grad_r = tf.placeholder(tf.float32, [None, 1])

        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
        self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

        self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)

        self.grad_theta = [tf.add(self.l2 * self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
        self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

        self.grad_norms = tf.global_norm(self.grad_theta)
        self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))

        self.sess.run(tf.global_variables_initializer())

    def __call__(self, state, action):
        feed_dict = {self.input_ph: self.feature_matrix.T[state, action]}
        with tf.get_default_session() as sess:
            sess.run(self.reward, feed_dict=feed_dict)

    def apply_grads(self, feat_map, grad_r):
        grad_r = np.reshape(grad_r, [-1, 1])
        feat_map = np.reshape(feat_map, [-1, self.dim_ss])
        _, grad_theta, l2_loss, grad_norms = self.sess.run(
            [self.optimize, self.grad_theta, self.l2_loss, self.grad_norms],
            feed_dict={self.grad_r: grad_r, self.input_ph: feat_map})
        return grad_theta, l2_loss, grad_norms

    def get_theta(self):
        return self.sess.run(self.theta)

    def get_rewards(self, states):
        rewards = self.sess.run(self.reward, feed_dict={self.input_ph: states})
        return rewards

    @memoize
    def phi(self, s, a):
        phi = np.zeros((self._dim_ss, 1), float)
        state = self._env.states[s]
        feature_ixs = range(self._dim_ss)
        if s in self._env.terminals or a == -1:
            return phi
        else:
            action = self._env.actions[a]
        for ix in feature_ixs:
            phi[ix] = self.features[ix](state, action)
        return phi
