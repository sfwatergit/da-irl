from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
# TODO:
# 1. Remove self._G and gamma in MDP since they are not part
# of problem definitions.
#
#
# modified from
# https://github.com/makokal/funzo/blob/master/funzo/models/mdp.py
import inspect
from abc import abstractmethod, abstractproperty, ABCMeta
from collections import Hashable
from functools import reduce
import numpy as np
import six
from cytoolz import memoize


class MDP(six.with_metaclass(ABCMeta)):
    def __init__(self, reward, transition, graph, gamma, env, terminals=None):
        self._G = graph
        self._reward = reward
        self._transition = transition
        self._gamma = gamma
        self._env = env
        self._terminals = terminals

    @property
    def env(self):
        return self._env

    @abstractproperty
    def S(self):
        """ Set of states in the MDP in an hashable container """
        raise NotImplementedError('Abstract property')

    @abstractproperty
    def A(self):
        """ Set of actions in the MDP in an hashable container """
        raise NotImplementedError('Abstract property')

    @abstractmethod
    def actions(self, state):
        """Set of available actions given the state"""
        raise NotImplementedError

    @property
    def reward(self):
        return self._reward

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
        return self._reward(state, action)

    @memoize
    def T(self, state, action):
        return self._transition(state, action)

    @property
    def terminals(self):
        return self._terminals

    def initial_state_distribution(self):
        """
        Return the initial state distribution of the MDP.
        """
        raise NotImplementedError


class TransitionFunction(six.with_metaclass(ABCMeta)):
    """ A MDP single step transition function

    .. math::

        T: \mathcal{S} \\times \mathcal{A} \\times \mathcal{S}
        \longmapsto \mathbb{R}

    A generic way of representing MDP transition operation for both discrete
    and continuous spaces. A T function simply takes and `action` at a given
    `state` and executes it based on the transition properties (which could
    include stochasticity, etc)

    Parameters
    -----------
    env : :class:`Domain` derivative object
        Object reference to the domain of the MDP that the controller is
        to be used on

    Attributes
    -----------
    env: :class:`Domain` derivative object
        Object reference to the domain of the underlying MDP

    """

    def __init__(self, env=None):
        self.env = env

    def __call__(seZlf, state, action, **kwargs):
        """ Execute the transition function

         Run the controller at `state` using `action` with optional parameters
         given in `kwargs`

         """
        raise NotImplementedError('Abstract method')


class State(six.with_metaclass(ABCMeta, Hashable)):
    """ MDP State

    A state in an MDP with all the relevant domain specific data. Such data
    is used in the reward function and transition function for computing
    various quantities. Every state must be a comparable object with an id

    """

    def __init__(self, state_id):
        super(State, self).__init__()
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
    """ MDP Action

    An action in an MDP with all the relevant domain specific data. Such data
    is used in the reward function and transition function for computing
    various quantities.

    """

    def __init__(self, action_id):
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


class RewardFunction(six.with_metaclass(ABCMeta)):
    def __init__(self, features, env=None, rmax=1.0):
        # keep a reference to parent MDP to get access to environment and
        # dynamics
        self._features = features
        self._env = env
        self._rmax = rmax
        self._dim_ss = reduce(lambda x, y: x + y, [len(x) for x in features])
        self._weights = np.random.normal(size=(1, self._dim_ss))
        self._feature_matrix = None
        self._r = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, **features):
        self._features += features

    @property
    def dim_ss(self):
        return self._dim_ss

    @abstractmethod
    def __call__(self, state, action):
        raise NotImplementedError

    @property
    def feature_matrix(self):
        if self._feature_matrix is None:
            self._feature_matrix = np.zeros((self._env.nS, self._env.nA, self.dim_ss), dtype=np.float32)
            for state in list(self._env.states.values()):
                actions = state.available_actions
                s = state.state_id
                for a in actions:
                    if s in self._env.terminals:
                        self._feature_matrix[s, a] = np.zeros(self._dim_ss, 1)
                    else:
                        self._feature_matrix[s, a] = self.phi(s, a).T
        return self._feature_matrix

    def phi(self, s, a):
        state = self._env.states[s]
        if s in self._env.terminals or a == -1:
            return np.zeros(self._dim_ss)
        else:
            action = self._env.actions[a]
        return np.concatenate([feature(state, action) for feature in self._features])


class LinearRewardFunction(six.with_metaclass(ABCMeta, RewardFunction)):
    """ RewardFunction using linear function approximation
    The reward funtion is define as,
    .. math::
        r(s, a) = \sum_i w_i \phi_i(s, a)
    where :math:`\phi_i(s, a)` is a feature defined over state and action
    spaces of the underlying MDP. The ``weights`` are the parameters of the
    model and are usually assumed to sum to 1 to ensure that the reward
    remains bounded, a typical assumption used in most RL planners.
    """

    _template = '_feature_'

    def __init__(self, weights, rmax=1.0, env=None):
        super(LinearRewardFunction, self).__init__(rmax, env)
        self._weights = np.asarray(weights)
        assert self._weights.ndim == 1, 'Weights must be 1D arrays'

    def update_parameters(self, **kwargs):
        """ Update the weights parameters of the reward function model """
        if 'reward' in kwargs:
            w = np.asarray(kwargs['reward'])
            assert w.shape == self._weights.shape, \
                'New weight array size must match reward function dimension'
            self._weights = w

    @property
    def kind(self):
        """ Type of reward function (e.g. tabular, LFA) """
        return 'LFA'

    @abstractmethod
    def phi(self, state, action):
        """ Evaluate the reward features for state-action pair """
        raise NotImplementedError('abstract')

    def __len__(self):
        """ Dimension of the reward function in the case of LFA """
        # - count all class members named '_feature_{x}'
        dim = 0
        for name in self.__class__.__dict__:
            item = getattr(self.__class__, name)
            if inspect.ismethod(item):
                if item.__name__.startswith(self._template):
                    dim += 1
        return dim


class FeatureExtractor(six.with_metaclass(ABCMeta)):
    def __init__(self, ident, size, **kwargs):
        self._size = size
        self.ident = ident
        if 'env' in kwargs:
            self.env = kwargs.pop('env')
        else:
            self.env = None

    @property
    def size(self):
        return self._size

    def __call__(self, state, action):
        """

        :param state: an agent state
        """
        raise NotImplementedError

    def __len__(self):
        return self._size

    def __str__(self):
        return "%s feature" % self.ident

    def __repr__(self):
        return self.__str__()