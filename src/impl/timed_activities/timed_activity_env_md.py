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
from impl.timed_activities.timed_activity_mdp import DurativeAction, \
    DurativeState
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
        self.home_goal_states = np.concatenate(
            [[s.state_id for s in self._states[('F H', 0)].values()],
             [s.state_id for s in self._states[('F H', 1)].values()]])
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
        self.state = self.home_start_state
        self.left_home = False
        return self.state_to_representation(self.state_id_map[self.state])

    def _step(self, action):
        """

        Args:
            action:

        Returns:

        """
        wrong_action_reward = [-999.0]

        # if not self._check_valid_action(action):
        #     p, next_state = self.mdp.T(self.state, action)[0]
        #     state = self.state_to_representation(next_state)
        #     return state, wrong_action_reward[0], True, {}

        p, next_state = self.mdp.T(self.state, action)[0]

        next_state_id = next_state.state_id

        next_fs = self.state_id_map[next_state_id]
        current_fs = self.state_id_map[self.state]

        fa = self.mdp.transition.action_id_map[action]
        finish_cond = current_fs.state_id in self.home_goal_states

        if finish_cond or p == 0:
            done = True
        else:
            done = False

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


class ExpertPatternSampler:
    def __init__(self, expert):
        self.expert = expert
        self.edf = pd.concat([create_hazard_df(
            build_persona_dataset(self.expert)).reset_index()])
        self.patterns = []
        self.pat_counts = self._get_expert_pattern_counts()
        self.pat_probs = self._get_expert_pattern_probs()

    def _get_expert_pattern_counts(self):
        for idx, group in self.edf.groupby('date').groups.items():
            dset = self.edf.iloc[group]
            pat = "".join(dset.symbol.values.tolist())
            self.patterns.append(pat)
        pat_counts = Counter(self.patterns)
        return pat_counts

    def _get_expert_pattern_probs(self):
        return np.asarray(list(self.pat_counts.values()), dtype=float) / sum(
            self.pat_counts.values())

    def sample_pattern(self):
        return list(self.pat_counts.keys())[
            np.random.choice(np.arange(len(self.pat_probs)), 1, replace=False,
                             p=self.pat_probs).tolist()[0]]

    def sample_df(self):
        pattern = self.sample_pattern()
        symbols = np.array([symbol for symbol in pattern])
        next_acts = np.roll(symbols, -1).astype('S16')
        prev_acts = np.roll(symbols, 1).astype('S16')
        prev_acts[0] = 'F H'
        prev_acts = prev_acts.astype(str)
        next_acts[-1] = 'S H'
        next_acts = next_acts.astype(str)
        stypes = []
        states = []
        trip_num = 0
        episodes = []
        for symbol in pattern:
            if symbol == '-':
                stypes.append('travel')
                trip_num += 1
                states.append('Trip {}'.format(trip_num))
            else:
                if trip_num == 0:
                    prefix = 'S'
                else:
                    prefix = trip_num
                stypes.append('activity')
                states.append('{} {}'.format(prefix, symbol))
            episodes.append('EP {}'.format(trip_num + 1))
        states[-1] = 'F H'
        sample_df = pd.DataFrame(
            {'symbol': symbols, 'next_act': next_acts, 'prev_act': prev_acts,
             'stype': stypes, 'state': states, 'episode': episodes})
        sample_df['time'] = np.nan
        sample_df['time_entry'] = np.nan
        sample_df['duration_prev'] = np.nan
        sample_df['duration_next'] = np.nan
        sample_df['time_budget'] = np.nan
        sample_df.loc[0, 'time'] = 0
        sample_df.loc[0, 'time_entry'] = 0
        sample_df.loc[0, 'time_budget'] = 1440
        sample_df.loc[0, 'duration_prev'] = 0
        sample_df.loc[len(sample_df) - 1, 'time_budget'] = 0
        return sample_df


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

class StackObservationWrapper(Wrapper):
    """
    This wrapper "stacks" `count` many consecutive observations together,
    i.e. it concatenates them along a new dimension.
    For time steps when not enough observations have already happened, the remaining
    space in the observation if filled by repeating the initial state.
    Currently only works for Box spaces.
    """
    def __init__(self, env, count, axis=0):
        """
        :param gym.Env env: The environment to wrap.
        :param int count: Number of observations that should be stacked.
        :param int axis: Axis along which to stack the values.
        """
        super(StackObservationWrapper, self).__init__(env)
        self._observations = deque(maxlen=count)
        self._axis = axis
        low = [0,0,0]
        high = [space.n for space in env.observation_space.spaces]
        low = np.stack([low]*count, axis=axis)
        high = np.stack([high]*count, axis=axis)
        self.observation_space = spaces.Tuple(low, high)

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._observations.append(obs)

        return np.stack(self._observations, axis=self._axis), rew, done, info

    def _reset(self):
        obs = self.env.reset()
        for i in range(self._observations.maxlen):
            self._observations.append(obs)

        return np.stack(self._observations, axis=self._axis)