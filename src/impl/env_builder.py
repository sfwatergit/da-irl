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

import six

from src.impl.activity_config import ATPConfig
from src.impl.activity_env import ActivityEnv
from src.impl.activity_mdp import ActivityState, TravelState, ATPAction, \
    ATPMDP, DeterministicTransition
from src.impl.activity_model import PersonModel
from src.impl.activity_rewards import ATPRewardFunction
from src.util.mandatory_activity_utils import get_mandatory_activities_done, \
    maybe_increment_mad


########################################################################

class AbstractStateBuilder(six.with_metaclass(ABCMeta)):
    # TODO: Pull abstract methods out of the initial implementation and into
    #       this class.

    def __init__(self, agent_id, horizon, interval_length, person_model):
        """A modular ``StateBuilder`` class for the ``ActivityEnv`` designed
        to allow different potential dynamics to define the ``Env``'s
        transition kernel.

        Args:
            agent_id (str): Unique identifier for the person
            horizon (int): Day discretized_horizon (used to discretize day)
            interval_length (int): Length of discretization interval
            person_model (PersonModel): A `PersonModel` representing one
            `ExpertPersona`.
        Attr:
            discretized_horizon (int): Horizon discretized by
                                     ``interval_length``.
        """
        self.person_model = person_model
        self.discretized_horizon = horizon / interval_length
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
    def run(self):
        """Returns a generator that populates states and a state graph.
        """
        raise NotImplementedError


class StateBuilder(AbstractStateBuilder):
    def __init__(self, agent_id, horizon, interval_length, person_model):
        # type: (str, int, int, PersonModel) -> None
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
                           ``discretized_horizon`` - 1 (corresponding to day
                           time
                           24:00 - discretization interval length (in minutes).
            interval_length (int): Length of discretization interval
            person_model (PersonModel): A `PersonModel` representing one
                                       `ExpertPersona`.
        """
        super(StateBuilder, self).__init__(agent_id, horizon,
                                           interval_length, person_model)
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
        if current_time_index == 0 or current_time_index == \
                self.discretized_horizon:
            return [self.home_activity_symbol]
        elif current_time_index == 1 or current_time_index == \
                self.discretized_horizon - 1:
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
        if next_time_index >= self.discretized_horizon:
            return [self.home_activity_symbol]
        elif next_time_index == self.discretized_horizon - 1:
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

    def run(self):
        """Returns a generator that populates an environments' states and
        state graph.
        """
        state_id = 0
        state_graph = {}
        states = {}
        # Step backward through time from discretized_horizon to 0
        for time_index in range(self.discretized_horizon, -1, -1):
            state_graph[time_index] = {}
            for symbol, next_symbols in self._get_states(time_index):
                for mad in get_mandatory_activities_done(
                        self.person_model, time_index):
                    # check prevents overwriting dict at
                    # state_graph.g[time_index][symbol]
                    if symbol not in state_graph[time_index]:
                        state_graph[time_index][symbol] = {}
                    State = ActivityState if symbol in self.activity_symbols \
                        else TravelState
                    state = State(state_id, symbol, time_index, mad,
                                  next_symbols)
                    state_graph[time_index][symbol][str(mad)] = state
                    states[state_id] = state
                    state_id += 1
        return states, state_graph


########################################################################


class AbstractTransitionBuilder(six.with_metaclass(ABCMeta)):
    def __init__(self, household_model, horizon, interval_length):
        """A modular transition builder object that is designed to create a
        generator for transitions from states in an ``ActivityEnv``.

        Args:
            horizon (int): This is the maximum index of time used to
                           discretize the agent's day. It is assumed that the
                           day starts at index 0 and continues through index
                           ``discretized_horizon`` - 1 (corresponding to day
                           time
                           24:00 - discretization interval length (in minutes).
            household_model (HouseholdModel): A `PersonModel` representing one
                          `ExpertPersona`.

        Attr:
            discretized_horizon (int): Horizon discretized by time unit
                                       interval length.
        """
        self.discretized_horizon = horizon / interval_length
        self.household_model = household_model


class TransitionBuilder(AbstractTransitionBuilder):
    def __init__(self, person_model, horizon, interval_length):
        """An initial implementation of the ``AbstractTransitionBuilder``
        """
        super(TransitionBuilder, self).__init__(
            person_model, horizon, interval_length)

    def get_transitions(self, state, person_model):
        """Compute the reachable next states characterizing the transition
        kernel.

        Args:
            state (ActivityState): The current state.

        Yields:
            tuple[int,str,nd.array]: The components of a transition.
        """
        for next_activity_symbol in state.reachable_symbols:
            current_mad = state.mad.astype(bool)
            if state.time_index + 1 < self.discretized_horizon:
                next_time_index = state.time_index + 1
            elif state.time_index + 1 <= self.discretized_horizon \
                    + 1:
                next_time_index = self.discretized_horizon
            else:
                raise ValueError
            if next_activity_symbol in \
                    person_model.mandatory_activity_set:
                next_mad = maybe_increment_mad(person_model, current_mad,
                                               next_activity_symbol)
            else:
                next_mad = current_mad.astype(int)
            yield next_time_index, next_activity_symbol, next_mad


########################################################################


class AgentBuilder(object):
    def __init__(self, config, person_model):
        """Builds an individual agent with MDP.

        Args:
            person_model (PersonModel): PersonModel used to build MDP.
            config (ATPConfig):  configuration parameters for agents.

        """
        self.person_model = person_model
        self.config = config

    def build_mdp(self, env):
        """Build the agent's MDP in the target environment.

        Populates the environment with states and resolves dependencies
        between environment and MDP.

        Args:
            env (ActivityEnv): Reference to environment.

        Returns:
            mdp (ATPMDP): The fully built MDP
        """
        states, state_graph = self._build_states()
        self._build_transitions(states, state_graph)
        actions = self._build_actions()
        env.add_states(states)
        env.update_G(state_graph)
        env.add_actions(actions)
        reward_function = ATPRewardFunction(self.config,
                                            self.person_model, env)
        transition_function = DeterministicTransition()
        mdp = ATPMDP(self.person_model, reward_function,
                     transition_function,
                     actions, states,
                     state_graph, self.config.irl_params.gamma)
        env.transition_matrix = mdp.transition_matrix

        return mdp

    def _build_states(self):
        """Build an ActivityMDP according to the supplied PersonModel.

        Returns:
            tuple[dict,dict]: The states and the populated state_graph.

        """
        state_builder = StateBuilder(self.person_model.agent_id,
                                     self.config.irl_params.horizon,
                                     self.config.profile_params.interval_length,
                                     self.person_model,
                                     )
        states, state_graph = state_builder.run()
        return states, state_graph

    def _build_actions(self):
        """Builds actions for the person (which serve as labels for
        transitions).

        Returns:
            tuple[dict,dict]: the mapping from action_ids to actions.

        """
        unique_symbols = set()
        unique_symbols |= (set(self.person_model.travel_models.keys()) |
                           set(self.person_model.activity_models.keys()))
        actions = {}
        for action_ix, el in enumerate(unique_symbols):
            actions[action_ix] = ATPAction(action_ix, el)
        return actions

    def _build_transitions(self, states, state_graph):
        transition_builder = TransitionBuilder(self.person_model,
                                               self.config.irl_params.horizon,
                                               self.config.profile_params.interval_length)
        for state in states.values():
            for next_time_index, next_activity, next_mad in \
                    transition_builder.get_transitions(state,
                                                       self.person_model):
                state.next_states.append(state_graph[next_time_index][
                                             next_activity][str(next_mad)])


class AbstractEnvBuilder(six.with_metaclass(ABCMeta)):

    @abstractmethod
    def _initialize_env(self):
        """Hook to run any initialization of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def add_agent(self, agent_builder):
        """Runs the ``StateBuilder`` and the ``TransitionBuilder``.

        Args:
            agent_builder (AgentBuilder):  An agent to
                                           instantiate within the env
        """
        raise NotImplementedError

    @abstractmethod
    def finalize_env(self):
        """Hook to run any finalization of the environment.

        """
        raise NotImplementedError


class HouseholdEnvBuilder(AbstractEnvBuilder):
    def __init__(self, household_model, interval_length, horizon):
        """Builds an ``ActivityEnv`` that considers intra-household
        decision-making.

        Args:
            household_model (HouseholdModel): Full set of configuration
                                             parameters for ActivityEnv IRL.
            horizon (int): Time horizon defining interval.
            interval_length (int): Duration of discretization interval.
        """
        self.interval_length = interval_length
        self.horizon = horizon
        self.household_model = household_model
        self._initialize_env()

    def _initialize_env(self):
        # Instantiate an empty environment:
        self.env = ActivityEnv()

    def add_agent(self, agent_builder):
        """Main method for running all components of the ``EnvBuilder``
        cycle.

        Permits instantiating multiple agent MDPs in the environment.
        The role of this method is to ensure that the dynamics cohere.

        Returns:
            (ActivityEnv): Fully initialized environment with all relevant
            properties available for use by algorithmic framework.
        """

        # Build states for each individual in the household
        # (independently of each other):
        self.env.mdps.append(agent_builder.build_mdp(self.env))

    def finalize_env(self):
        """A few additional tasks to complete for the sake of convenience.

        This method assumes that the state graph has already been built for all
        agents in the mdp.
        """
        # initialize environment action_space and state_space
        self.env.dim_A = len(self.env.actions)
        self.env.dim_S = len(self.env.states)
        # set some other useful properties
        self.env.horizon = self.horizon
        self.env.interval_length = self.interval_length

        # Assume same start and possible terminal activities for all agents
        # in household.
        self.env.home_activity = \
            self.household_model.home_activity_symbols[0]
        self.env.home_start_state = \
            self.env.g[0][self.env.home_activity].values()[0]
        self.env.home_goal_states.extend(
            self.env.g[self.horizon / self.interval_length]
            [self.env.home_activity].values())
        return self.env


########################################################################

if __name__ == '__main__':
    config_file = sys.argv[1]
    with open(config_file, 'r') as fp:
        config = ATPConfig(data=json.load(fp))
    env_builder = HouseholdEnvBuilder(config)

    env = env_builder.add_agent()
    print('done')
