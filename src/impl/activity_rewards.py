import os
import os
import platform

import matplotlib
import numpy as np
import tensorflow as tf

from src.core.mdp import RewardFunction
from src.impl.activity_features import ActivityFeature, \
    create_act_at_x_features, TravelFeature
from src.util.math_utils import get_subclass_list, cartesian, \
    create_dir_if_not_exists
from src.util.tf_utils import fc_net, get_session

if platform.system() == 'Darwin':
    matplotlib.rcParams['backend'] = 'agg'
else:
    matplotlib.rcParams['backend'] = 'Agg'

import matplotlib.pyplot as plt

plt.interactive(False)


class ATPRewardFunction(RewardFunction):
    """
    """

    def __init__(self, config, person_model, agent_id, env, rmax=1.0,
                 opt_params=None,
                 nn_params=None, initial_theta=None):
        """Computes the activity reward based on the state which is the current
        activity and time of day.

        Initialized with config-defined scoring parameters.

        Args:
            config (ATPConfig): System configuration parameters.
            rmax (float): Maximum value of the reward (not currently used)
            opt_params (dict[str,obj]):
            nn_params (dict[str,obj]):
            initial_theta (nd.array):
        """
        self.env = env
        self.agent_id = agent_id
        self.config = config
        self.person_model = person_model
        self.activity_features = self.make_activity_features()
        self.trip_features = self.make_travel_features()
        self._make_indices()

        super(ATPRewardFunction, self).__init__(
            self.activity_features + self.trip_features, rmax=rmax,
            initial_weights=initial_theta, env=env)

        self.name = "reward_{}".format(person_model.agent_id)
        with tf.variable_scope(self.name):
            self._init(opt_params, nn_params, initial_theta)
            self.scope = tf.get_variable_scope().name

    def _init(self, opt_params, nn_params, initial_theta):
        """

        Args:
            opt_params:
            nn_params:
            initial_theta:
        """
        if nn_params is None:
            nn_params = {'h_dim': 32, 'reg_dim': 10, 'name': 'maxent_irl'}

        if opt_params is None:
            opt_params = {'lr': 0.3}

        self.lr = opt_params['lr']

        self.h_dim = nn_params['h_dim']
        self.reg_dim = nn_params['reg_dim']

        self.input_size = self.dim_phi

        self.input_ph = tf.placeholder(tf.float32,
                                       shape=[None, self.input_size],
                                       name='dim_phi')

        reward = fc_net(self.input_ph, n_layers=1, dim_hidden=self.h_dim,
                        out_act=None,
                        init=initial_theta, name=self.name)

        self.theta = reward.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)

        self.reward = reward

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        self.grad_r = tf.placeholder(tf.float32, [None, 1])

        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
        self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

        self.grad_theta = tf.gradients(self.reward, self.theta, - self.grad_r)

        self.grad_theta = [
            tf.add(self.reg_dim * self.grad_l2[i], self.grad_theta[i]) for i in
            range(len(self.grad_l2))]
        # self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

        self.grad_norms = tf.global_norm(self.grad_theta)
        self.optimize = self.optimizer.apply_gradients(
            zip(self.grad_theta, self.theta))

        self.sess = get_session()
        self.sess.run(tf.global_variables_initializer())

    def make_travel_features(self):
        """Make features for accessible trip modes for the agent.
        Args:
            env (src.impl.activity_env.ActivityEnv):

        Returns:

        """
        return [i(mode, self.person_model,
                  self.config.profile_params.interval_length,
                  self.agent_id, env=self.env) for i in
                get_subclass_list(TravelFeature) for mode in
                self.person_model.travel_models.keys()]

    def make_activity_features(self):
        """Make features for each important activity for the agent.
        """
        activity_features = [i(self.person_model,
                               self.config.profile_params.interval_length,
                               self.agent_id,
                               env=self.env) for
                             i in get_subclass_list(ActivityFeature)]

        acts = [activity[0] for activity in
                self.person_model.activity_groups.values()]

        time_range = np.arange(0, self.config.irl_params.horizon,
                               self.config.profile_params.interval_length)

        prod = cartesian([acts, time_range])

        act_at_x_features = [
            create_act_at_x_features(where, when,
                                     self.config.profile_params.interval_length,
                                     self.person_model, self.agent_id)(
                env=self.env)
            for where, when in prod]

        activity_features += act_at_x_features
        return activity_features

    def _make_indices(self):
        """

        """
        assert (len(self.activity_features) > 0) and (
                len(self.trip_features) > 0)
        self._activity_feature_ixs = range(len(self.activity_features))
        self._trip_feature_ixs = range(len(self.activity_features),
                                       len(self.activity_features) +
                                       len(self.trip_features))

    def __call__(self, state, action):
        """

        Args:
            state:
            action:
        """
        self.phi(state, action)

    def phi(self, state, action):
        """

        Args:
            state:
            action:

        Returns:

        """
        phi = np.zeros((self._dim_phi, 1), float)
        feature_ixs = range(self._dim_phi)
        for ix in feature_ixs:
            phi[ix] = self.features[ix](state, action)
        return phi

    def apply_grads(self, grad_r):
        """

        Args:
            grad_r:

        Returns:

        """
        feat_map = self.feature_matrix
        grad_r = np.reshape(grad_r, [-1, 1])
        feat_map = np.reshape(feat_map, [-1, self.dim_phi])
        _, grad_theta, l2_loss, grad_norms = self.sess.run(
            [self.optimize, self.grad_theta, self.l2_loss, self.grad_norms],
            feed_dict={self.grad_r: grad_r, self.input_ph: feat_map})
        return grad_theta, l2_loss, grad_norms

    def get_theta(self):
        """

        Returns:

        """
        return self.sess.run(self.theta)[0]

    def get_trainable_variables(self):
        """

        Returns:

        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_rewards(self):
        """

        Returns:

        """
        feed_dict = {
            self.input_ph: self.feature_matrix.reshape([-1, self.dim_phi])}
        rewards = self.sess.run(self.reward, feed_dict)
        return rewards.reshape([self._env.dim_S, self._env.dim_A])

    def plot_current_theta(self, ident):
        from src.impl.activity_model import ActivityModel
        """

        Args:
            ident:
        """
        image_path = os.path.join(self.config.general_params.log_dir,
                                  os.path.join("expert_{}".format(ident)),
                                  self.config.general_params.images_dir)
        create_dir_if_not_exists(image_path)

        for activity in self.person_model.activity_groups.values():
            activity = activity[0]  # type: ActivityModel
            activity_interval = []
            for feature in self.activity_features:  # type: ActivityFeature
                if feature.ident.startswith(activity.site_type):
                    activity_interval.append(self.activity_features.index(
                        feature))
            plot_theta(self.get_theta(), activity_interval,
                       name=' '.join(activity.site_type.split(
                           '_')).capitalize(),
                       disc_len=self._env.interval_length,
                       image_path=image_path,
                       ident=ident)


def plot_theta(theta, activity_interval, disc_len, name, image_path,
               show=False, ident='', color='r'):
    """

    Args:
        theta:
        home_int:
        work_int:
        other_int:
        disc_len:
        image_path:
        show:
        ident:
    """
    feats = theta[activity_interval]
    plot_reward(feats, disc_len, image_path, name, color, show, ident)
    plt.clf()


def plot_reward(ys, disc_len=15., image_path='', name='', color='b',
                show=False, ident=''):
    """

    Args:
        ys:
        disc_len:
        image_path:
        name:
        color:
        show:
        ident:
    """
    xs = np.arange(0, len(ys)) * float(disc_len) / 60
    plt.plot(xs, ys, color)
    plt.title('Marginal Utility vs. Time of Day for {} Activity'.format(name))
    plt.xlabel('time (hr)')
    plt.ylabel('marginal utility (utils/hr)')
    if show:
        plt.show()
    else:
        plt.savefig(image_path + '/{}_activity_persona_{}'.format(name, ident))
