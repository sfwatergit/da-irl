from abc import abstractproperty, ABCMeta

import os
import six

from src.algos.maxent_irl import MaxEntIRL
from src.impl.activity_config import ATPConfig
from src.impl.activity_env import ActivityEnv
from src.misc import logger
from src.util.math_utils import create_dir_if_not_exists


class ExpertAgent(six.with_metaclass(ABCMeta)):
    def __init__(self, config, env, learning_algorithm):
        # type: (ATPConfig, ActivityEnv, MaxEntIRL) -> ExpertAgent
        """PersonaAgent representation.

        Args:
            config (ATPConfig): parameters specifications for this expert agent.
            env (ActivityEnv):
            learning_algorithm (MaxEntIRL):
        """
        self.env = env
        self._config = config
        self._learning_algorithm = learning_algorithm

    @abstractproperty
    def identifier(self):
        raise NotImplementedError('Expert agent must have an identifier')

    @abstractproperty
    def trajectories(self):
        raise NotImplementedError('Must implement trajectories method')

    def learn_reward(self):
        self._learning_algorithm.train(self.trajectories)
        logger.save_itr_params(self.env.irl_params.num_iters,
                               self._learning_algorithm
                               .get_itr_snapshot(self.env.irl_params.num_iters)
                               .update({'agent': self.identifier}))
        self._learning_algorithm.reward.plot_current_theta(self.identifier)

    @property
    def reward_function(self):
        return self._learning_algorithm.reward

    @property
    def policy(self):
        return self._learning_algorithm.policy
