import os
import os
import platform
import sys
import matplotlib
import numpy as np

from src.core.mdp import TFRewardFunction
from src.impl.activity_features import ActivityFeature, \
    create_act_at_x_features, TravelFeature
from src.util.math_utils import get_subclass_list, cartesian, \
    create_dir_if_not_exists

if platform.system() == 'Darwin':
    matplotlib.rcParams['backend'] = 'agg'
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.interactive(False)


class ATPRewardFunction(TFRewardFunction):
    """
    """

    def __init__(self, config, person_model, agent_id, env, rmax=1.0,
                 opt_params=None):
        self.agent_id=agent_id
        self.config = config
        self.person_model = person_model
        self.env = env
        self.activity_features = self.make_activity_features()
        self.trip_features = self.make_travel_features()
        self._make_indices()
        features = self.activity_features+self.trip_features
        super(ATPRewardFunction, self).__init__(env, rmax,
                                                opt_params, agent_id=agent_id,
        features=features)

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
        #
        # act_at_x_features = [
        #     create_act_at_x_features(where, when,
        #                              self.config.profile_params.interval_length,
        #                              self.person_model, self.agent_id)(
        #         env=self.env)
        #     for where, when in prod]
        #
        # activity_features += act_at_x_features
        return activity_features

    @property
    def feature_matrix(self):
        """Compute the feature matrix, \Phi for each state and action pair
        in the domain.

        Returns:
            nd.array[float]: |\mathcal{S}| X |\mathcal{A}| x |\phi| dimensioned
                            matrix of features.
        """
        if self._feature_matrix is None:
            self._feature_matrix = np.zeros(
                (self.env.dim_S, self.env.dim_A, self.dim_phi),
                dtype=np.float32)
            for state in list(self.env.states.values()):
                action_ids = [self.env.reverse_action_map[
                                  self.env.states[next_state].symbol]
                              for next_state in self.env.G.successors(
                        state.state_id)]
                s = state.state_id
                for a in action_ids:
                    self._feature_matrix[s, a] = self.phi(state,
                                                          self.env.actions[
                                                              a]).T
        return self._feature_matrix

    def _make_indices(self):
        assert (len(self.activity_features) > 0) and (
                len(self.trip_features) > 0)
        self._activity_feature_ixs = range(len(self.activity_features))
        self._trip_feature_ixs = range(len(self.activity_features),
                                       len(self.activity_features) +
                                       len(self.trip_features))

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
            plot_theta(self.theta(), activity_interval,
                       name=' '.join(activity.site_type.split(
                           '_')).capitalize(),
                       disc_len=self.env.interval_length,
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
