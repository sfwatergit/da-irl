from abc import abstractproperty

from src.algos.maxent_irl import MaxEntIRL
from src.impl.activity_config import ATPConfig
from src.impl.activity_env import ActivityEnv


class ExpertAgent(object):
    def __init__(self, config, env, learning_algorithm):
        # type: (ATPConfig, ActivityEnv, MaxEntIRL) -> ExpertAgent
        """PersonaAgent representation.

        Args:
            config (ATPConfig): parameters specifications for this expert agent.
            env (ActivityEnv):
            learning_algorithm (MaxEntIRL):
            persona (Persona): persona representation of traveler.
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

    @property
    def reward_function(self):
        return self._learning_algorithm.reward

    @property
    def policy(self):
        return self._learning_algorithm.policy


