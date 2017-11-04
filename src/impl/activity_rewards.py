from src.core.mdp import RewardFunction
from src.impl.activity_features import ActivityFeature, create_act_at_x_features, TripFeature
from src.misc import tf_utils
from src.misc.math_utils import get_subclass_list, cartesian
import tensorflow as tf
import numpy as np

from cytoolz import memoize


class ActivityRewardFunction(RewardFunction):
    """
    Computes the activity reward based on the state which is the current activity and time of day.
    Initialized with config-defined scoring parameters.
    """

    def __init__(self, params, env):
        self.activity_features = [i(params, env=env) for i in get_subclass_list(ActivityFeature)]
        acts = [env.home_act, env.work_act, env.other_act]
        time_range = np.arange(0, 1800, 15)
        prod = cartesian([acts, time_range])
        act_at_x_features = [create_act_at_x_features(where, when, 15, params)(env=env) for where, when in prod]
        self.activity_features.extend(act_at_x_features)
        self.trip_features = []
        for mode in params.travel_params.keys():
            self.trip_features.extend([i(mode, params, env=env) for i in get_subclass_list(TripFeature)])
        features = []
        features.extend(self.activity_features)
        features.extend(self.trip_features)
        self._activity_feature_ixs = xrange(len(self.activity_features))
        self._trip_feature_ixs = xrange(len(self.activity_features),
                                        len(self.activity_features) + len(params.travel_params.keys()))
        super(ActivityRewardFunction, self).__init__(features, env)

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


class ActivityNNReward(RewardFunction):
    def __init__(self, params, opt_params=None, nn_params=None, env=None):

        if nn_params is None:
            nn_params = {'h1': 100, 'h2': 50, 'l2': 10, 'name': 'maxent_deep_irl'}

        if opt_params is None:
            opt_params = {'lr': 0.01}

        self.lr = opt_params['lr']

        self.activity_features = [i(params, env=env) for i in get_subclass_list(ActivityFeature)]
        acts = [env.home_act, env.work_act, env.other_act]
        time_range = np.arange(0, 1800, 15)
        prod = cartesian([acts, time_range])
        act_at_x_features = [create_act_at_x_features(where, when, 15, params)(env=env) for where, when in prod]
        self.activity_features.extend(act_at_x_features)
        self.trip_features = []
        for mode in params.travel_params.keys():
            self.trip_features.extend([i(mode, params, env=env) for i in get_subclass_list(TripFeature)])
        features = []
        features.extend(self.activity_features)
        features.extend(self.trip_features)
        self._activity_feature_ixs = xrange(len(self.activity_features))
        self._trip_feature_ixs = xrange(len(self.activity_features),
                                        len(self.activity_features) + len(params.travel_params.keys()))
        super(ActivityNNReward, self).__init__(features, env)

        self.sess = tf.Session()

        self.name = nn_params['name']
        self.h1 = nn_params['h1']
        self.h2 = nn_params['h2']
        self.l2 = nn_params['l2']

        input_size = self.dim_ss

        self.input_ph = tf.placeholder(tf.float32, [None, input_size])

        with tf.variable_scope(self.name):
            fc1 = tf_utils.fc(self.input_ph, self.h1, scope="fc1", activation_fn=tf.nn.elu,
                              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            fc2 = tf_utils.fc(fc1, self.h2, scope="fc2", activation_fn=tf.nn.elu,
                              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            reward = tf_utils.fc(fc2, 1, scope="reward")
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        self.reward = reward
        self.theta = theta

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        self.grad_r = tf.placeholder(tf.float32, [None, 1])

        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
        self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

        self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)

        self.grad_theta = [tf.add(self.l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
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
        _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms],
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