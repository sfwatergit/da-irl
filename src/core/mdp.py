"""MDP Base class representation

Modified from
https://github.com/makokal/funzo/blob/master/funzo/models/mdp.py
"""
# Py3 Compat:
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

# Default
from abc import abstractmethod, abstractproperty, ABCMeta
from collections import Hashable
from functools import reduce

# Third Party:
import numpy as np
import six


class MDP(six.with_metaclass(ABCMeta)):
    def __init__(self, reward_function, transition, gamma):
        """ Markov Decision Problem (MDP) model

       For general MDPs, states and action can be continuous making it
       hard to efficiently represent them using standard data structures. In the
       case of discrete MDPs, it is straightforward to use array of comparable
       objects to represent the states.

       In the continuous cases, we assume that only a sample of the state and
       action spaces will be used, and these can also be represented a simple
       hashable data structure (indexed by state or action ids).

       .. note:: This design deliberately leaves out the details of
       **states** and **actions** to be handled by the domain object which
       includes a reference to an MDP object. Additionally, transitions and
       reward which are in general functions are represented as separate
       *callable* objects with references to relevant data needed. This
       allows a unified interface for both **discrete** and **continuous**
       MDPs and further extensions

       Args:
           reward_function (RewardFunction)
               Reward function for the MDP with all the relevant parameters
           transition  (TransitionFunction)
               Represents the transition function for the MDP. All transition
               relevant details such as stochasticity are handled therein.
           gamma (float): MDP discount factor in the range [0, 1)

        """
        self._reward_function = reward_function
        self._transition = transition
        self._gamma = gamma
        self._terminals = None

    @abstractproperty
    def S(self):
        """ Set of states in the MDP in an hashable container """
        raise NotImplementedError('Abstract property')

    @abstractproperty
    def A(self):
        """ Set of actions in the MDP in an hashable container """
        raise NotImplementedError('Abstract property')

    def T(self, state, action):
        """ Transition kernel"""
        return self._transition(state, action)

    @abstractmethod
    def available_actions(self, state):
        """Set of available actions given the state"""
        raise NotImplementedError('Abstract method')

    @property
    def reward_function(self):
        return self._reward_function

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        """ MDP Discount factor """
        if 0.0 > value >= 1.0:
            raise ValueError('MDP `discount` must be in [0, 1)')
        self._gamma = value

    def R(self, state, action):
        return self._reward_function(state, action)

    @property
    def terminals(self):
        if self._terminals is None:
            self._terminals = self._env.terminals
        return self._terminals


########################################################################


class TransitionFunction(six.with_metaclass(ABCMeta)):

    def __init__(self, env=None):
        """ A MDP single step transition function

        .. math::

            T: \mathcal{S} \\times \mathcal{A} \\times \mathcal{S}
            \longmapsto \mathbb{R}

        A generic way of representing MDP transition operation for both
        discrete and continuous spaces. A transition function simply takes an
        `action` at a given `state` and executes it based on the transition
         properties (which could include stochasticity, etc).

        Args:
            env  (ActivityEnv): derivative object ``Env`` reference to the
                                environment of the MDP that the controller is
                                to be used on.
        """
        self._env = env

    def __call__(self, state, action, **kwargs):
        """ Execute the transition function

         Run the controller at `state` using `action` with optional parameters
         given in `kwargs`

        Args:
            state (State):

         """
        raise NotImplementedError('Abstract method')


########################################################################


class State(six.with_metaclass(ABCMeta, Hashable)):
    def __init__(self, state_id):
        """ MDP State

        A state in an MDP with all the relevant domain specific data. Such data
        is used in the reward function and transition function for computing
        various quantities. Every state must be a comparable object with an id.

        Args:
            state_id (int): Unique integer index for the state.
        """
        self._state_id = state_id

    @abstractmethod
    def __hash__(self):
        """ State hash function """
        raise ValueError('Implement a hash function')

    @property
    def state_id(self):
        """ State unique identifier """
        return self._state_id

    @abstractmethod
    def __eq__(self, other):
        """ State comparator function """
        raise NotImplementedError('Implement equality of states')


class Action(six.with_metaclass(ABCMeta, Hashable)):
    def __init__(self, action_id):
        """ MDP Action

        An action in an MDP with all the relevant domain specific data. Such
        data is used in the reward function and transition function for
        computing various quantities. Every action must be a comparable
        object with an id.

        Args:
            action_id (int): Unique integer index for the action.
        """
        super(Action, self).__init__()
        self._action_id = action_id

    @abstractmethod
    def __hash__(self):
        """ Action hash function """
        raise ValueError('Implement a hash function')

    @property
    def action_id(self):
        """ Action unique identifier """
        return self._action_id

    @abstractmethod
    def __eq__(self, other):
        """ Action comparator function """
        raise NotImplementedError('Implement equality of actions')


########################################################################


class RewardFunction(six.with_metaclass(ABCMeta)):

    def __init__(self, features, env=None, rmax=1.0, initial_weights=None):
        """ MDP reward function model

        Rewards are as functions of state and action spaces of MDPs, i.e.

        .. math::
            r: \mathcal{S} \times \mathcal{A} \longmapsto \mathbb{R}

        Rewards for a state and action pair, :math:`r(s, a)` are accessed via
        the ``__call__`` method while appropriate reward function parameters are
        set in the constructor. In the :class:`MDP` object, this function
        will be called via :class:`MDP.R` function.

        Args:
            features (list[FeatureExtractor]): Features making up \phi
            env (ActivityEnv): Reference to parent environment
            rmax (float): Reward upper bound.
            initial_weights (nd.array[float]): Initial value of weights by which
                                               to multiply feature functions.
        """
        self._features = features
        self._env = env
        self._rmax = rmax
        self._dim_phi = None
        self._update_dim_phi()
        self._feature_matrix = None
        self._r = None

    def _update_dim_phi(self):
        self._dim_phi = reduce(lambda x, y: x + y, [len(x) for x in
                                                    self._features])

    @abstractmethod
    def __call__(self, state, action):
        """ Evaluate the reward function for the (state, action) pair
        """
        raise NotImplementedError('Abstract method')

    @property
    def features(self):
        """All of the feature extractors, {\phi}, for this reward function.

        Returns:
            (FeatureExtractor): List of features.
        """
        return self._features

    @features.setter
    def features(self, **features):
        """Sets feature extractors, {\phi}, for this reward function

        Args:
            **features (FeatureExtractor): List of features.
        """
        self._features += features
        self._update_dim_phi()

    @property
    def dim_phi(self):
        """ Dimension of the reward function"""
        return self._dim_phi

    def __len__(self):
        """ Dimension of the reward function"""
        return self._dim_phi

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
                (self._env.dim_S, self._env.dim_A, self.dim_phi),
                dtype=np.float32)
            for state in list(self._env.states.values()):
                action_ids = [self._env.reverse_action_map[next_state.symbol]
                              for next_state in state.next_states]
                s = state.state_id
                for a in action_ids:
                    if s in self._env.terminals:
                        self._feature_matrix[s, a] = np.zeros([self._dim_phi])
                    else:
                        self._feature_matrix[s, a] = self.phi(state,
                                                              self._env.actions[
                                                                  a]).T
        return self._feature_matrix

    def phi(self, state_id, action_id):
        """Computes the feature vector at a state and action.

        If the state is a goal state, then an array of zeros is returned. The
        dimensions of the vector correspond to the number of features per
        state-action pair.

        Args:
            state_id (int): Unique state identifier
            action_id (int): Unique action identifier

        Returns:
            nd.array[float]: |\phi| X 1 array of values corresponding to a
                            row of the feature matrix for the given state
                            and action.
        """
        raise NotImplementedError('Abstract method must be implemented')


########################################################################

class FeatureExtractor(six.with_metaclass(ABCMeta)):
    def __init__(self, name, size, **kwargs):
        """Extracts features given domain-specific representations of states
        and actions.

        Args:
            name (str): Identifier for this extractor
            size (int): Dimensions of the feature function
        """
        self._size = size
        self.ident = name
        if 'env' in kwargs:
            self.env = kwargs.pop('env')
        else:
            self.env = None

    @property
    def size(self):
        """Size of the feature.

        Returns:
            (int) returns the size of this feature in the feature space.
        """
        return self._size

    @abstractmethod
    def __call__(self, state, action):
        """Extracts features for a given state and action pairing.

        Args:
            state (State): The state for which to extract the feature.
            action (Action): The action for which to extract the feature.
        """
        raise NotImplementedError

    def __len__(self):
        return self._size

    def __str__(self):
        return "%s feature" % self.ident

    def __repr__(self):
        return self.__str__()
