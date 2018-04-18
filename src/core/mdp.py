"""MDP Base class representation

Modified from
https://github.com/makokal/funzo/blob/master/funzo/models/mdp.py
"""
# Py3 Compat:
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

# Default
from abc import abstractmethod, abstractproperty, ABCMeta
from builtins import *
from collections import Hashable
from functools import reduce

# Third Party:
import numpy as np
import six
import tensorflow as tf

from util.tf_utils import fc_net, get_session


class MDP(six.with_metaclass(ABCMeta)):
    def __init__(self, reward_function, transition, gamma, terminals=None):
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
        self.transition = transition
        self._gamma = gamma
        self._terminals = terminals

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
        return self.transition(state, action)

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
            self._terminals = self.T._env.terminals
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
            features (nd.array[FeatureExtractor]): Features making up \phi
            env (ActivityEnv): Reference to parent environment
            rmax (float): Reward upper bound.
            initial_weights (nd.array[float]): Initial value of weights by which
                                               to multiply feature functions.
        """
        self._features = features
        self.env = env
        self._rmax = rmax
        self._dim_phi = None
        self._update_dim_phi()

    def _update_dim_phi(self):
        self._dim_phi = reduce(lambda x, y: x + y, [len(x) for x in
                                                    self._features])

    def __call__(self, state, action):
        """ Evaluate the reward function for the (state, action) pair
        """
        self.phi(state, action)

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

    def phi(self, state, action):
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
        phi = np.zeros((self._dim_phi, 1), float)
        feature_ixs = range(self._dim_phi)
        for ix in feature_ixs:
            phi[ix] = self.features[ix](state, action)
        return phi


class LinearRewardFunction(RewardFunction):
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

    def __init__(self, weights, rmax=1.0, domain=None):
        super().__init__(rmax, domain)
        self._weights = np.asarray(weights)
        # assert self._weights.ndim == 1, 'Weights must be 1D arrays'

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

    def phi(self, state, action):
        """ Evaluate the reward features for state-action pair """
        pass

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


class TFRewardFunction(RewardFunction):

    def __init__(self, env=None, rmax=1.0,
                 opt_params=None,
                 agent_id=None,
                 nn_params=None, initial_theta=None, features=None):
        """
           Initialized with config-defined scoring parameters.

        Args:
            config (ATPConfig): System configuration parameters.
            rmax (float): Maximum value of the reward (not currently used)
            opt_params (dict[str,obj]):
            nn_params (dict[str,obj]):
            initial_theta (nd.array):
        """
        self.rmax = rmax
        self._feature_matrix = None
        self.name = "reward_{}".format(agent_id)
        super(TFRewardFunction, self).__init__(env=env, features=features)

        with tf.variable_scope(self.name):
            self.scope = tf.get_variable_scope().name

            if nn_params is None:
                nn_params = {'h_dim': 32, 'reg_dim': 5, 'name': 'maxent_irl'}

            if opt_params is None:
                opt_params = {'lr': 0.1}

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

            self._theta = reward.graph.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES)

            self.reward = reward

            self.optimizer = tf.train.AdamOptimizer(self.lr)

            self.grad_r = tf.placeholder(tf.float32, [None, 1])

            self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self._theta])
            self.grad_l2 = tf.gradients(self.l2_loss, self._theta)

            self.grad_theta = tf.gradients(self.reward, self._theta,
                                           - self.grad_r)

            self.grad_theta = [
                tf.add(self.reg_dim * self.grad_l2[i], self.grad_theta[i]) for i
                in
                range(len(self.grad_l2))]
            self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

            self.grad_norms = tf.global_norm(self.grad_theta)
            self.optimize = self.optimizer.apply_gradients(
                zip(self.grad_theta, self._theta))

            self.sess = get_session()
            self.sess.run(tf.global_variables_initializer())

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
                                                          self.env.actions[a]).T
        return self._feature_matrix

    @feature_matrix.setter
    def feature_matrix(self,fm):
        if self._feature_matrix is None:
            self._feature_matrix = fm


    @property
    def theta(self):
        """

        Returns:

        """
        return self.sess.run(self._theta)[0]

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
        return rewards.reshape([self.env.dim_S, self.env.dim_A])


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
