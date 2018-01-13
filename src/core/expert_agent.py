import os.path as osp
from abc import abstractproperty, ABCMeta

import six

from src.algos.maxent_irl import MaxEntIRL
from src.impl.activity_env import ActivityEnv
from src.misc import logger


class ExpertAgent(six.with_metaclass(ABCMeta)):
    def __init__(self, config, person_model, mdp, learning_algorithm):
        """PersonaAgent representation.

        Args:
            config (ATPConfig): General configuration parameters.
            person_model (PersonModel): activity and travel specifications for
            this expert agent.
            mdp (ActivityMDP): The MDP for this agent.
            learning_algorithm (IRLAlgorithm):
        """
        self.config = config
        self.mdp = mdp
        self._person_model = person_model
        self._learning_algorithm = learning_algorithm

    @abstractproperty
    def identifier(self):
        raise NotImplementedError('Expert agent must have an identifier')

    @abstractproperty
    def trajectories(self):
        raise NotImplementedError('Must implement trajectories method')

    def learn_reward(self, skip_policy=0, iterations=-1):
        prefix = "pid: %s | " % self.identifier
        tabular_log_file_pr = osp.join(self.config.general_params.log_dir,
                                       osp.join(
                                           "expert_{}".format(self.identifier),
                                           self.config.tabular_log_file))
        logger.add_tabular_output(tabular_log_file_pr)
        logger.push_prefix(prefix)
        logger.push_tabular_prefix(prefix)

        if iterations == -1:
            iterations = self.config.irl_params.num_iters
        self._learning_algorithm.train(self.trajectories, iterations,
                                       skip_policy)
        params = self._learning_algorithm.get_itr_snapshot(
            self.config.irl_params.num_iters)
        params.update({'agent': self.identifier})
        logger.save_itr_params(self.config.irl_params.num_iters, params)
        self._learning_algorithm.reward.plot_current_theta(self.identifier)
        logger.pop_prefix()
        logger.pop_tabular_prefix()
        logger.remove_tabular_output(tabular_log_file_pr)

    @property
    def reward_function(self):
        return self._learning_algorithm.reward

    @property
    def policy(self):
        return self._learning_algorithm.policy
