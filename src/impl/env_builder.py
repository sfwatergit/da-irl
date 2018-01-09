"""Builds an environment's state space and transition kernel according to the
following general algorithm:

```
   env = ActivityEnv()
   state_id = 0
   for each household:
       for each person in household:
           state_builder = StateBuilder(person)
           for each time_slice:
               for each activity in reachable_activities:
                   state = state_builder.run(activity, time_slice, mad)
                   env.states[state_id] = state
                   state_id+=1
   transition_builder = TransitionBuilder(...)
   for each state in env.states:
       for next_activity in state.reachable_activities:
           next_state = TransitionBuilder.run(next_activity, ...)
           state.next_states.append(next_state)
```

Designed with modularity in mind so that StateBuilder and TransitionBuilder
components can be extended and/or switched out.
"""
import json
import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import combinations

import numpy as np
import six

from src.impl.activity_config import ATPConfig
from src.impl.activity_env import ActivityEnv
from src.impl.activity_mdp import ActivityState, TravelState, ATPAction
from src.impl.activity_model import PersonModel
from src.util.math_utils import to_onehot


class MandatoryActivityMixin(object):
    def __init__(self, person_model):
        self.person_model = person_model
        self.mandatory_activity_set = person_model.mandatory_activity_set
        self.mandatory_activity_map = OrderedDict(
            (v, to_onehot(k, len(self.mandatory_activity_set)))
            for k, v in enumerate(self.mandatory_activity_set))

    def _get_mandatory_activities_done(self, current_time_index):
        """Compute the possible mandatory activities that have already been
        completed at the __start__ of this timeslice (i.e., prior to the
        current time index).

        The result is a one-hot vector keyed to the set of
        mandatory activities (sorted lexicographically).

        Conditional logic respects the following boundary conditions:
            1. Agent cannot start off with any mandatory activities completed.
            2. It is impossible for an agent to have completed any mandatory
            activities by the time they've reached the second timeslice
            (travel would've been required).

        Args:
            current_time_index (int): The current timeslice index.

        Yields:
            nd.array[str]: A one-hot vector of mandatory activities completed
                          (or an array of all zeros if none have been
                          completed).
        """
        if current_time_index < 2:
            yield np.zeros(len(self.mandatory_activity_set), dtype=int)
        else:
            num_possible_ma = min(current_time_index,
                                  len(self.mandatory_activity_set))
            for i in range(num_possible_ma, -1, -1):
                possible_mad = combinations(self.mandatory_activity_set, i)
                if i == 0:
                    yield np.zeros(len(self.mandatory_activity_set), dtype=int)
                else:
                    for ma in possible_mad:
                        yield np.array(sum([self.mandatory_activity_map[a] for a
                                            in ma]))

    def _maybe_increment_mad(self, current_mad, next_activity_symbol):
        """Utility function to compute the possible next set of completed
        mandatory activities given a reachable next state symbol.

        Args:
            current_mad (nd.array[int]): One-hot vector of the current
                                        completed maintenance activities.
            next_activity_symbol (str): Symbol indicative of the next
                                        activity or travel state for which to
                                        compute the next completed
                                        maintenance activities.

        Returns:
            (nd.array[int]): One-hot vector of the projected completed
                             maintenance activities.

        """
        return current_mad.astype(int) + (
                self.mandatory_activity_map[next_activity_symbol].astype(
                    bool) & ~current_mad).astype(int)


class AbstractStateBuilder(six.with_metaclass(ABCMeta)):
    # TODO: Pull abstract methods out of the initial implementation and into
    #       this class.

    # TODO: Implement the modules as mixins (multiple inheritance). This would
    #       allow for something like mandatory activities to be plugged in
    #       and out  of the state generator (run method).
    def __init__(self, agent_id, horizon, person_model):
        """A modular ``StateBuilder`` class for the ``ActivityEnv`` designed
        to allow different potential dynamics to define the ``Env``'s
        transition kernel.

        Args:
             agent_id (str): Unique identifier for the person
            horizon (int): Day horizon (used to discretize day)
            person_model (PersonModel): A `PersonModel` representing one
            `ExpertPersona`.
        """
        self.person_model = person_model
        self.horizon = horizon
        self.agent_id = agent_id

    @abstractmethod
    def _get_states(self, time_index):
        """Generate the components of the reachable states at the current time
        index.

        Args:
            time_index (int): current time index.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, env):
        """Returns a generator that populates an environments' states and
        state graph.

        Args:
            env (ActivityEnv): An ``ActivityEnv``, which may have an empty or
                               partially filled state graph.
        """
        raise NotImplementedError


class StateBuilder(AbstractStateBuilder, MandatoryActivityMixin):
    def __init__(self, agent_id, horizon, person_model):
        # type: (str, int, PersonModel) -> None
        """Builds the state dynamics in an ``Env`` for an individual
        ``ExpertPersona`` agent.

        This concrete implementation of ``StateBuilder`` serves as the
        default implementation for a single agent's daily activity-travel
        states. State symbols are referenced by the agent's identifier like
        so:

        'H#{1}' (home activity for agent with agent_id '1')

        *** Planned Implementation *********************************************
        If an agent has a joint activity with another agent,
        it will be indicated using symbols as follows:

        `H#{1,3}` (home activity for two agents with agent_ids '1' and '3'
        ************************************************************************

        Attr:
            work_activity_symbol (str): the primary work activity for the
                                        individual
            home_activity_symbol (str): the primary home activity for the
                                        individual
            mandatory_activity_set (str): the set of mandatory activities
                                        that the agent must perform each day.



        Args:
            agent_id (str): Unique identifier for the person
            horizon (int): This is the maximum index of time used to
                           discretize the agent's day. It is assumed that the
                           day starts at index 0 and continues through index
                           ``horizon`` - 1 (corresponding to day time
                           24:00 - discretization interval length (in minutes).
            person_model (PersonModel): A `PersonModel` representing one
                                       `ExpertPersona`.
        """
        super(StateBuilder, self).__init__(agent_id, horizon, person_model)

        self.work_activity_symbol = person_model.work_activity.symbol
        self.home_activity_symbol = person_model.home_activity.symbol
        self.activity_symbols = person_model.activity_models.keys()
        self.travel_mode_symbols = person_model.travel_models.keys()
        self.all_symbols = self.activity_symbols + self.travel_mode_symbols

    def _get_possible_symbols(self, current_time_index):
        # type: (int) -> list[str]
        """Computes, at any given timeslice, the state symbol indicators.

        Args:
            current_time_index (int): The index of the current time slice
            (indicating the beginning of the time interval).

        Returns:
            (list[str]): A list of state symbols.
        """
        if current_time_index == 0 or current_time_index == self.horizon:
            return [self.home_activity_symbol]
        elif current_time_index == 1 or current_time_index == self.horizon - 1:
            return [self.home_activity_symbol] + self.travel_mode_symbols
        else:
            return self.all_symbols

    def _get_next_reachable_symbols(self, current_symbol, current_time_index):
        """Used to simplify working with ``TransitionBuilder`` by
        pre-populating the reachable next activity symbols (modulo the
        timeslice).

        If in an activity state, the agent can either stay in the activity or
        transition to a travel state (any available mode).

        If in a travelling state, the agent can either stay in the current
        mode, transition to a different mode, or arrive at an activity
        (essentially, any transition is available).

        Conditional logic respects the following boundary conditions:
            1. If it is the last timeslice of the day, then the agent must
            be at home.
            2. If it is the next-to-last timeslice of the day, then the
            agent may only be traveling (to home) or already at home.

        Args:
            current_symbol (str): the current state type of the agent.
            current_time_index (int): start_time of activity
                                  or travel segment represented by symbol
                                  used mainly to double check constraints on
                                  activity start/end times.

        Returns:
            (list): A list of the available legal actions for the current state.

        """
        next_time_index = current_time_index + 1
        if next_time_index >= self.horizon:
            return [self.home_activity_symbol]
        elif next_time_index == self.horizon - 1:
            return [self.home_activity_symbol] + self.travel_mode_symbols
        else:
            if current_symbol in self.activity_symbols:
                res = [current_symbol] + self.travel_mode_symbols
            elif current_symbol in self.travel_mode_symbols:
                res = self.all_symbols
            else:
                raise ValueError("%s not Found!" % current_symbol)
            return res

    def _get_states(self, time_index):
        """Generate the components of the reachable states at the current time
        index.

        Args:
            time_index (int): The current time index.
        """
        for symbol in self._get_possible_symbols(time_index):
            yield symbol, self._get_next_reachable_symbols(symbol,
                                                           time_index)

    def run(self, env):
        """Returns a generator that populates an environments' states and
        state graph.

       Args:
           env (ActivityEnv): An ``ActivityEnv``, which may have an
           empty or partially filled state graph.
        """
        state_id = 0 if len(env.states) == 0 else env.states.keys()[-1]
        # Step backward through time from horizon to 0
        for time_index in range(self.horizon, -1, -1):
            env.g[time_index] = {}
            for symbol, next_symbols in self._get_states(time_index):

                for mad in self._get_mandatory_activities_done(time_index):
                    # check prevents overwriting dict at
                    # env.g[time_index][symbol]
                    if symbol not in env.g[time_index]:
                        env.g[time_index][symbol] = {}
                    State = ActivityState if symbol in self.activity_symbols \
                        else \
                        TravelState
                    state = State(state_id, symbol, time_index, mad,
                                  next_symbols)
                    env.g[time_index][symbol][str(mad)] = state
                    env.states[state_id] = state
                    state_id += 1
        # Additional initialization of environment properties


class AbstractTransitionBuilder(six.with_metaclass(ABCMeta)):
    def __init__(self, person_model, horizon):
        """A modular transition builder object that is designed to create a
        generator for transitions from states in an ``ActivityEnv``.

        Args:
            horizon (int): This is the maximum index of time used to
                           discretize the agent's day. It is assumed that the
                           day starts at index 0 and continues through index
                           ``horizon`` - 1 (corresponding to day time
                           24:00 - discretization interval length (in minutes).
            person_model (PersonModel): A `PersonModel` representing one
                          `ExpertPersona`.
        """
        self.horizon = horizon
        self.mandatory_activity_set = person_model.mandatory_activity_set


class TransitionBuilder(AbstractTransitionBuilder, MandatoryActivityMixin):
    def __init__(self, person_model, horizon):
        """An initial implementation of the ``AbstractTransitionBuilder``
        """
        super(TransitionBuilder, self).__init__(person_model, horizon)

    def get_transitions(self, state):
        """Compute the reachable next states characterizing the transition
        kernel.

        Args:
            state (ActivityState): The current state.

        Yields:
            tuple[int,str,nd.array]: The components of a transition.
        """
        for next_activity_symbol in state.reachable_symbols:
            current_mad = state.mad.astype(bool)
            if state.time_index + 1 < self.horizon:
                next_time_index = state.time_index + 1
            elif state.time_index + 1 <= self.horizon + 1:
                next_time_index = self.horizon
            else:
                raise ValueError
            if next_activity_symbol in self.mandatory_activity_set:
                next_mad = self._maybe_increment_mad(current_mad,
                                                     next_activity_symbol)
            else:
                next_mad = current_mad.astype(int)
            yield next_time_index, next_activity_symbol, next_mad


class AbstractEnvBuilder(six.with_metaclass(ABCMeta)):
    # XXXX: There should really only be one env builder with StateBuilder
    # and TransitionBuilder provided as mixins or declarative metaclasses.
    def __init__(self, config):
        """Base class for environment dynamics builder.

        Args:
            config (ATPConfig): General configuration parameters for
                                ActivityEnv IRL
        """
        self.config = config

    @abstractmethod
    def run(self):
        """Runs the ``StateBuilder`` and the ``ActivityBuilder``.

        Returns:
            (ActivityEnv): The ``Env`` with the dynamics fully built.
        """
        raise NotImplementedError


class HouseholdEnvBuilder(AbstractEnvBuilder):
    def __init__(self, atp_config):
        # type: (ATPConfig) -> None
        """Builds an ``ActivityEnv`` that considers intra-household
        decision-making.

        Args:
            atp_config (ATPConfig): Full set of configuration parameters for
                                    ActivityEnv IRL.
        """
        super(HouseholdEnvBuilder, self).__init__(atp_config)
        self.household_model = atp_config.household_params.household_model

    def run(self):
        """Main method for running all components of the ``EnvBuilder``
        cycle.

        Returns:
            (ActivityEnv): Fully initialized environment with all relevant
            properties available for use by algorithmic framework.
        """
        # Instantiate an empty environment:
        env = ActivityEnv()

        # Build states for each individual in the household
        # (independently of each other):
        for agent_id, person_model in \
                self.household_model.household_member_models.items():
            state_builder = StateBuilder(agent_id, config.irl_params.horizon,
                                         person_model)
            state_builder.run(env)

        # Transitions governed by already built state graph nodes
        for agent_id, person_model in \
                self.household_model.household_member_models.items():
            # FIXME: It would be preferable not to need to have a separate
            #        TransitionBuilder for each PersonModel
            transition_builder = TransitionBuilder(person_model,
                                                   config.irl_params.horizon)

            for state in env.states.values():
                for next_time_index, next_activity, next_mad in \
                        transition_builder.get_transitions(state):
                    state.next_states.append(
                        env.g[next_time_index][next_activity][str(next_mad)])

        # Initialize other properties of environment here.
        self._finialize_env_init(env)
        return env

    def _finialize_env_init(self, env):
        """A few additional tasks to complete for the sake of convenience.

        This method assumes that the state graph has already been built.

        Args:
            env (ActivityEnv): The environment to finalize.
        """
        # initialize actions:
        self._build_actions(env)
        # initialize environment action_space and state_space
        env.nA = len(env.actions)
        env.nS = len(env.states)
        # set some other useful properties
        env.horizon = self.config.irl_params.horizon
        # TODO: Either create profile builder config for household and/or ensure
        # profile builder params are identical b/w agents.
        env.segment_minutes = int(
            self.config.profile_params.SEQUENCES_RESOLUTION.strip('min'))
        # XXXX: For now, just use first available symbol for
        # home_activity. Later, remember this must eventually be done for
        # potentially multiple agents.
        env.home_activity = \
            self.config.household_params.household_model.home_activity_symbols[
                0]
        env.home_start_state = env.g[0][env.home_activity].values()[0]
        env.home_goal_states.extend(
            env.g[env.horizon][env.home_activity].values())

    def _build_actions(self, env):
        """Builds actions in the env (which serve as labels for transitions).

        Args:
            env (ActivityEnv): an activity-travel environment
        """
        unique_symbols = set()
        for person_model in \
                self.config.household_params.household_model \
                        .household_member_models \
                        .values():
            unique_symbols |= (set(person_model.travel_models.keys()) |
                               set(person_model.activity_models.keys()))
        env.actions = {}
        for action_ix, el in enumerate(unique_symbols):
            env.actions[action_ix] = ATPAction(action_ix, el)
        env._action_rev_map = dict(
            (v.succ_ix, k) for k, v in env.actions.items())


if __name__ == '__main__':
    config_file = sys.argv[1]
    with open(config_file, 'r') as fp:
        config = ATPConfig(data=json.load(fp))
    env_builder = HouseholdEnvBuilder(config)
    env = env_builder.run()
    print('done')
