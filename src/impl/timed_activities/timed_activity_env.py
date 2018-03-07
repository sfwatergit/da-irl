from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from collections import Counter

import numpy as np
import pandas as pd
from gym import Env
from gym.envs.registration import EnvSpec
from rllab.core.serializable import Serializable

from rllab.misc import logger
from gym.spaces import MultiDiscrete, Discrete


from impl.timed_activities.hazard_utils import create_hazard_df, \
    build_persona_dataset
from impl.timed_activities.timed_activity_mdp import DurativeAction
from src.impl.timed_activities.timed_activity_mdp import TimedActivityMDP, \
    Representation


class TimedActivityEnv(Env):

    def __init__(self, expert_pattern_sampler, mdp, *args, **kwargs):
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

        super().__init__(*args, **kwargs)
        self.pattern_sampler = expert_pattern_sampler
        self.mdp = mdp
        self.representation = Representation(mdp)
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

        self.current_pattern = self.pattern_sampler.sample_pattern()
        self.home_activity = self.current_pattern[0]
        self.home_start_state = self._states[('S H', 0)][0].state_id
        self.home_goal_states = [list(self._states[('F H', 1)].values())[
                                     0].state_id]
        self.terminals = self.home_goal_states

        self.state = self.home_start_state


        self.g = None

        self.__observation_space = None

        self._transition_matrix = None
        self._reward_function = None

    @property
    def horizon(self):
        return self._horizon

    def _check_valid_action(self, action):
        fa = self.mdp.transition.action_id_map[action]  # type: DurativeAction

        current_fs = self.mdp.transition.state_id_map[self.state]  # type:
        # DurativeState
        # Rules:

        # 0. Can stay at home if at home at end of day
        if current_fs.state_id in self.home_goal_states and fa.duration == 0:
            return True

        # 1. Can only move to the next state if done:

        is_travelling = True if (current_fs.symbol, current_fs.is_done) in \
                                self.mdp.travel_states else False
        wants_to_travel = True if fa.next_state_symbol in \
                                  self.mdp.travel_actions else False
        if current_fs.is_done:
            # 2. Can only go from travel to activity and vice versa
            if is_travelling:
                if not wants_to_travel:
                    return True
                else:
                    return False
            # Not traveling... can only go to travel
            else:
                if wants_to_travel:
                    if 'S H' in current_fs.symbol:
                        self.left_home = True
                    return True
                else:
                    return False
        # Not done... can only stay in same kind of state
        else:
            if is_travelling:
                if wants_to_travel:
                    return True
                else:
                    return False
            else:
                if wants_to_travel:
                    return False
                else:
                    return True

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
            self.__observation_space = Discrete(
                len(self.mdp.transition.state_id_map))
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
        self.current_pattern = self.pattern_sampler.sample_pattern()
        self.state = self.home_start_state
        self.left_home = False
        return self.state

    def _step(self, action):
        """

        Args:
            action:

        Returns:

        """
        wrong_action_reward = [-999.0]

        if not self._check_valid_action(action):
            return self.state, wrong_action_reward[0], True, {}

        p, next_state = self.mdp.T(self.state, action)[0]

        next_state_id = next_state.state_id

        fs = self.mdp.transition.state_id_map[next_state_id]

        if self.left_home and 'S H' in fs.symbol:
            return self.state, wrong_action_reward[0], True, {}

        fa = self.mdp.transition.action_id_map[action]

        if next_state_id in self.home_goal_states or p == 0 or \
                fs.time_index == self.horizon:
            done = True
        else:
            done = False

        self.state = next_state_id

        if p == 0:
            reward = wrong_action_reward
        else:
            reward = self.mdp.R(fs, fa)
        extras = {}
        return self.state, reward[0], done, extras

    def _reward(self, state, action):
        """
        r: S,A -> \mathbb{R}
        Returns: reward for state and action

        """
        return self.reward_function(state, action)

    def get_home_action_id(self):
        assert self.actions is not None
        for id, act in self.actions.items():
            if self.home_activity == act.next_state_symbol:
                return id

    def available_actions(self, state):
        state = self.mdp.transition.state_id_map[state]
        available_actions = self.mdp.available_actions(state)
        act_reprs = [action.action_id for
                     action in available_actions]
        return act_reprs

    def get_param_values(self):
        return None

    def log_diagnostics(self, paths):
        # Ntraj = len(paths)
        # acts = np.array([traj['actions'] for traj in paths])
        obs = np.array([np.sum(traj['observations'], axis=0) for traj in paths])

        state_count = np.sum(obs, axis=0)
        # state_count = np.mean(state_count, axis=0)
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
