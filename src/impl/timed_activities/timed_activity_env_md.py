from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from collections import Counter, deque

import numpy as np
import pandas as pd
from gym import Env, spaces, Wrapper
from gym.envs.registration import EnvSpec
from gym.spaces import Discrete, Tuple
from rllab.misc import logger

from impl.timed_activities.hazard_utils import create_hazard_df, \
    build_persona_dataset
from impl.timed_activities.timed_activity_mdp import DurativeState
from src.impl.timed_activities.timed_activity_mdp import TimedActivityMDP, \
    Representation


class TimedActivityEnv(Env):

    def __init__(self, mdp, *args, **kwargs):
        """Activity-Travel environment representing the planning domain with
        which activity-travel planning agent interact.

        This environment (:py:class:`~gym.Env`) represents a daily
        activity-travel pattern domain.

        During episodes, agents take actions
        (:py:class:`da-irl.src.impl.mdp.ATPAction) by calling the step method on
        the current Env. Actions represent a choice of next activity or trip and
        the time  to spend at the current activity). The agent then observes the
        state (:py:class:`da-irl.src.impl.mdp.ATPState) of the environment
        change in  response to its action as well as a reward immediately upon
        registering his or her choice. That is,  the agent receives an update in
        the form of next_state, reward.

        Once the agent has reached its home state and no longer wishes to make
        trips during the day, the agent then transitions to its home_goal_state,
        an absorbing state in the Markov chain describing the daily activity
        pattern.

        It is expected that the environment is initialized per a configurable
        ``EnvBuilder``. All non-derivative properties are defined per the
        functionality of the builder object.

        Args:
            mdp (TimedActivityMDP):

        """

        super().__init__()
        self.mdp = mdp
        self._representation = Representation(mdp)
        self.spec = EnvSpec('timed_activity_env-v0')

        self._horizon = self.mdp.horizon
        self.interval_length = self.mdp.interval_length

        self._dim_A = len(self.mdp.A)
        self._dim_S = len(self.mdp.S)

        self.__action_space = None
        self._actions = mdp.actions
        self._reverse_action_map = None

        self._states = mdp.states
        self._actions = mdp.actions

        # convenience delegations
        self.state_id_map = mdp.transition.state_id_map
        self.action_id_map = mdp.transition.action_id_map

        self.home_start_state = self._states[('S H', 0)][0].state_id
        self.home_goal_states = [s.state_id for s in self.state_id_map.values()
                                 if s.time_index == self._horizon and ('H' in
                                            s.symbol and '=>' not in s.symbol)]
        self.terminals = self.home_goal_states

        self.max_len = 12

        self.state = self.home_start_state

        self.g = None

        self.__observation_space = None

        self._transition_matrix = None
        self._reward_function = None

    def representation_to_state(self, state_rep):
        return self._representation.representation_to_state(state_rep)

    def state_to_representation(self, state):
        return self._representation.state_to_representation(state)

    @property
    def horizon(self):
        return self._horizon

    def step(self, action):
        return self._step(action)

    def reset(self):
        return self._reset()

    def render(self, mode='human'):
        pass

    @property
    def dim_A(self):
        if self._dim_A is None:
            self._dim_A = len(self.actions)
        return self._dim_A

    @property
    def dim_S(self):
        if self._dim_S is None:
            self._dim_S = len(self.states)
        return self._dim_S

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    def sample_action(self, state):
        acts = self.available_actions(state)
        return acts[np.random.choice(np.arange(len(acts)))]

    @property
    def action_space(self):
        if self.__action_space is None:
            self.__action_space = Discrete(
                len(self.mdp.transition.action_id_map))
        return self.__action_space

    @property
    def observation_space(self):
        if self.__observation_space is None:
            self.__observation_space = Tuple([
                Discrete(self._representation.dim_activities),
                Discrete(int(self.horizon / self.interval_length) + 1),
                Discrete(2)])
        return self.__observation_space

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, transition_matrix):
        self._transition_matrix = transition_matrix

    @property
    def reward_function(self):
        return self._reward_function

    @reward_function.setter
    def reward_function(self, reward_function):
        self._reward_function = reward_function

    def _reset(self):
        self.current_step = 0
        self.state = self.home_start_state
        self.left_home = False
        return self.state_to_representation(self.state_id_map[self.state])

    def _step(self, action):
        """

        Args:
            action:

        Returns:

        """
        wrong_action_reward = [-1.0]

        # if not self._check_valid_action(action):
        #     p, next_state = self.mdp.T(self.state, action)[0]
        #     state = self.state_to_representation(next_state)
        #     return state, wrong_action_reward[0], True, {}

        p, next_state = self.mdp.T(self.state, action)[0]

        next_state_id = next_state.state_id

        next_fs = self.state_id_map[next_state_id]
        current_fs = self.state_id_map[self.state]

        fa = self.mdp.transition.action_id_map[action]
        finish_cond = next_fs.state_id in self.home_goal_states

        if finish_cond or p == 0:
            done = True
        else:
            done = False
            self.current_step += 1

        self.state = next_state_id
        if finish_cond and p != 0:
            reward = [0.0]
        elif p == 0:
            reward = wrong_action_reward
        else:
            reward = [0.0]
            # reward = self.mdp.R(current_fs, fa)
        extras = {}
        if reward == wrong_action_reward:
            state = self.state_to_representation(current_fs)
        else:
            state = self.state_to_representation(self.state_id_map[self.state])
        return state, reward[0], done, extras

    def _reward(self, state, action):
        """
        r: S,A -> \mathbb{R}
        Returns: reward for state and action

        """
        return self.reward_function(state, action)

    def available_actions(self, state):
        available_actions = self.mdp.available_actions(state)
        act_reprs = [action.action_id for action in available_actions]
        return act_reprs

    def get_param_values(self):
        return None

    def log_diagnostics(self, paths):

        obs = np.array([np.sum(traj['observations'], axis=0) for traj in paths])

        state_count = np.sum(obs, axis=0)

        state_freq = state_count / float(np.sum(state_count))
        for state in range(self.dim_S):
            if state_freq[state] > 0:
                logger.record_tabular('AvgStateFreq%d' % state,
                                      state_freq[state])

    def _seed(self):
        return 122

def build_expert_patterns(experts, env):
    """

    Args:
        experts ():
        env ():

    Returns:
        dict:
    """
    eps = []
    for path in experts:
        states,actions = path['observations'],path['actions']
        ep = []
        for s in states[1:]:
            state = env.representation_to_state(s)
            sym = state.symbol
            if '=>' in sym:
                ep.append('-')
            else:
                ep.append(sym[-1])
        eps.append(''.join(ep))
    eps_counts = Counter(eps)
    tot = sum(eps_counts.values())
    exp_val_dict = {k:np.round((v/tot),3) for k,v in eps_counts.items()}
    exp_val_dict = sorted(exp_val_dict.items(), key=lambda x: x[1], reverse=True)
    return exp_val_dict


class StateBuilder:
    def __init__(self):
        self.state_index = 0

    def _reset_state_index(self):
        self.state_index = 0

    def inc_state_index(self):
        self.state_index += 1

    def build_states(self, symbol, start_time, end_time, reachable_states,
                     is_done):
        res = {}
        for time in range(start_time * 15, end_time * 15, 15):
            res[time] = DurativeState(self.state_index, symbol, time,
                                      reachable_states, is_done)
            self.inc_state_index()
        return res


class LimitDuration(object):
  """End episodes after specified number of steps."""

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    if self._step is None:
      raise RuntimeError('Must reset environment.')
    observ, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      self._step = None
    return observ, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()
