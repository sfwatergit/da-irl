from abc import ABCMeta, abstractmethod
from src.impl.activity_model import PersonModel
from src.impl.activity_config import ATPConfig
import six


class ExpertAgent(six.with_metaclass(ABCMeta)):
    """
    """

    def __init__(self,
                 config,
                 person_model,
                 mdp,
                 learning_algorithm,
                 trajectories,
                 identifier):
        """PersonaAgent representations.

        Args:
            config (ATPConfig): General configuration parameters.
            person_model (PersonModel): activity and travel specifications for
            this expert agent.
            mdp (ActivityMDP): The MDP for this agent.
            learning_algorithm (IRLAlgorithm): Algorithm used to learn the
                reward encoding the movement of the ``ExpertAgent``.
        """

        self._identifier = identifier
        self.config = config
        self.mdp = mdp
        self.person_model = person_model
        self._learning_algorithm = learning_algorithm
        self._trajectories = trajectories

    @property
    def identifier(self):
        """

        Returns:

        """
        return self._identifier

    @property
    def trajectories(self):
        """

        Returns:

        """
        return self._trajectories

    @abstractmethod
    def setup_learning(self):
        raise NotImplementedError('Must implement abstract method!')

    def learn_reward(self, skip_policy=0, iterations=-1):
        """Learn the reward function for the current agent.

        Args:
            skip_policy (int): number of iterations of the outer loop of
                learning to skip
            iterations (int): number of iterations for which to train IRL.
        """
        self.setup_learning()
        if iterations == -1:
            iterations = self.config.irl_params.num_iters
        self._learning_algorithm.train(self.trajectories, iterations,
                                       skip_policy)
        self.finish_learning()

    @abstractmethod
    def finish_learning(self):
        raise NotImplementedError('Must implement abstract method!')

    @property
    def reward_function(self):
        """

        Returns:
            src.impl.activity_rewards.ATPRewardFunction:
        """
        return self._learning_algorithm.reward

    @property
    def policy(self):
        """

        Returns:

        """
        return self._learning_algorithm.policy


class AbstractPathProcessor(six.with_metaclass(ABCMeta)):

    def __init__(self, mdp, person_model):
        """Processes paths in external encoding to IRL-compatible format.

        Args:
            mdp (T <= src.core.mdp.MDP): Dynamics of the IRL environment.
            person_model (src.impl.activity_model.PersonModel):
        """
        self._person_model = person_model
        self._mdp = mdp

    @abstractmethod
    def run(self, path_matrix):
        """

        Args:
            path_matrix (numpy.ndarray):
        """
        raise NotImplementedError('Must implement abstract method!')
