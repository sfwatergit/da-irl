import numpy as np

from src.core.mdp import State, TransitionFunction, MDP
from src.impl.activity_mdp import ATPAction
from util.math_utils import to_onehot, from_onehot


class DurativeState(State):
    def __init__(self, state_id, symbol, time_index, reachable_symbols,
                 is_done=False):
        super(DurativeState, self).__init__(state_id)
        self.symbol = symbol
        self.time_index = time_index
        self.next_states = []
        self.reachable_symbols = reachable_symbols
        self.is_done = is_done

    def __str__(self):
        if self.is_done:
            ast = '*'
        else:
            ast = ''
        return '{}:[{}, {}{}]'.format(self.state_id, self.time_index,
                                      self.symbol, ast)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.state_id, self.time_index))

    def __eq__(self, other):
        return self.state_id == other.state_id and self.time_index == \
               other.time_index


class DurativeAction(ATPAction):
    act_index = 0

    def __init__(self, next_state_symbol, duration):
        super(DurativeAction, self).__init__(DurativeAction.act_index,
                                             next_state_symbol)
        self.duration = duration
        DurativeAction._incr_act_index()

    @classmethod
    def reset_index(cls):
        cls.act_index = 0

    @classmethod
    def _incr_act_index(cls):
        cls.act_index += 1

    def __str__(self):
        if self.duration == 0.0:
            out_dur = '>'
        else:
            out_dur = self.duration
        return "{}:[{}, {}]".format(self.action_id, self._next_state_symbol,
                                    out_dur)

    def __hash__(self):
        return hash((self.next_state_symbol, self.duration))


class TimedActivityTransition(TransitionFunction):

    def __init__(self, states, actions, horizon):
        state_id_map = {}
        action_id_map = {}
        self.horizon = horizon
        for state_symbol in states.values():
            for state in state_symbol.values():
                state_id_map[state.state_id] = state
        for action_symbol in actions.values():
            for action in action_symbol.values():
                action_id_map[action.action_id] = action
        self.state_id_map = state_id_map
        self.action_id_map = action_id_map
        self.states = states
        self.actions = actions
        super(TimedActivityTransition, self).__init__()

    def __call__(self, state, action, **kwargs):
        """

        Returns:
            list[tuple[int, DurativeState]]: A list of tuples of the
            probability of the  each next state and the state itself. For
            deterministic environments, this will be a single number
            (0 if impossible and 1 if possible) and the state. The
            probabilities must sum to 1.
        """
        state = self.state_id_map[state]
        action = self.action_id_map[action]
        current_time = state.time_index
        duration = action.duration
        next_time_key = current_time + duration

        if next_time_key > self.horizon:
            return [(0, self.states[('F H', 0)][self.horizon])]

        if state.symbol == 'F H' and state.is_done == 1:
            return [(1, self.states[('F H', 1)][self.horizon])]
        next_symbol = action.next_state_symbol

        next_is_done = not state.is_done
        if "=>" in state.symbol:  # traveling already
            if '=>' not in next_symbol:
                # if state.symbol[-1] != next_symbol[-1]:
                return [(0, self.states[('F H', 0)][self.horizon])]
                # prefix = ""
                # next_symbol = next_symbol[-1]
                # trip = False
            else:  # travel
                # symbol is matching
                if state.symbol[-1] != next_symbol[-1] or action.duration == 0:
                    return [(0, self.states[('F H', 0)][self.horizon])]
                else:
                    prefix = ""
                    trip = False
                    next_symbol = next_symbol[-1]
                    next_is_done = False
        else:  # at activity
            if '=>' in next_symbol:  # wants to travel
                if state.is_done:
                    if action.duration == 0:
                        prefix = "Trip => "
                        next_symbol = next_symbol[-1]
                        trip = True
                    else:
                        return [(0, self.states[('F H', 0)][self.horizon])]
                else:
                    return [(0, self.states[('F H', 1)][self.horizon])]
            else:  # wants to stay at activity
                if state.is_done:
                    if next_time_key ==self.horizon and 'H' in next_symbol and \
                            'S' \
                            not in state.symbol and 'H' in state.symbol:  #
                        # can't
                        # stay if done
                        return [(1, self.states[('F H', 0)][self.horizon])]
                    else:
                        return [(0, self.states[('F H', 0)][self.horizon])]
                else:
                    if state.symbol[-1] != next_symbol or action.duration == 0:
                        return [(0, self.states[('F H', 0)][self.horizon])]
                    else:  # ok
                        prefix = ""
                        next_symbol = next_symbol[-1]
                        trip = False

        ep_num = self.episode_number(state, trip, next_is_done)

        next_symbol_key = "{}{} {}".format(prefix, ep_num, next_symbol)
        next_state_key = (next_symbol_key, next_is_done)

        if 'S' in state.symbol:
            if 'H' in next_symbol_key:
                if trip and next_time_key != 0 and not state.is_done:
                    return [(0, self.states[('F H', 1)][self.horizon])]
                if next_time_key == self.horizon:
                    return [(0, self.states[('F H', 0)][self.horizon])]
                if next_time_key == 0:
                    return [(0, self.states[('S H', 0)][0])]

        # bogus state key... punish!
        if 'S' in next_state_key[0] and 'H' not in next_state_key[0]:
            next_state_key = ('2 {}'.format(next_state_key[0][-1]))
            if next_time_key > self.horizon:
                next_time_key = self.horizon
            return [(0, self.states[(next_state_key, 0)][next_time_key])]
        # Episode is too high... fail!
        elif next_state_key not in self.states and ep_num > 0:
            return [(0, self.states[('F H', 1)][self.horizon])]
        if next_time_key not in self.states[next_state_key]:
            return [(0, self.states[('F H', 1)][self.horizon])]

        return [(1, self.states[next_state_key][next_time_key])]

    @staticmethod
    def episode_number(state, trip, next_is_done):
        # if state.symbol
        if state.symbol != 'S H' and state.symbol != 'F H':
            # if we're on a trip, then episode number is after '=> ':
            if '=>' in state.symbol:
                split_key = '=> '
                ep_num = int(state.symbol.split(split_key)[-1][0])
            else:
                split_key = ' '
                ep_num = int(state.symbol.split(split_key)[0])
        # if done, then transitioning to next episode
        elif state.symbol == 'S H':
            if trip and state.is_done:
                return 2
            else:
                return 'S'
        else:
            ep_num = ''
        if trip and not next_is_done:
            ep_num += 1
        return ep_num


class Representation:

    def __init__(self, mdp):
        self.mdp = mdp
        self.states = mdp.states
        self.interval_length = self.mdp.interval_length
        self.actions = mdp.actions
        self.state_map = dict([(v, k) for k, v in enumerate(np.unique([
            state.symbol
            for state in list(
                self.mdp.transition.state_id_map.values())]))])
        self.rev_state_map = dict([(v, k) for k, v in self.state_map.items()])
        self.action_map = dict([(v, k) for k, v in enumerate(
            np.unique([
                action.next_state_symbol
                for
                action in list(
                    self.mdp.transition.action_id_map.values())]))])
        self.rev_action_map = dict([(v, k) for k, v in self.action_map.items()])
        self.dim_activities = len(self.state_map)
        self.dim_actions = len(self.action_map)

    def state_to_representation(self, state):
        time_index = int(state.time_index / self.interval_length)
        activity_symbol = state.symbol
        is_done = state.is_done
        inds = np.array([self.state_map[activity_symbol], time_index,
                         is_done], dtype=np.int32)
        dims = [self.dim_activities, self.mdp.horizon, 2]
        return inds

    def action_to_representation(self, action):
        duration_ind = int(action.duration / self.interval_length)
        action_symbol_ind = self.action_map[action.next_state_symbol]
        inds = np.array([action_symbol_ind, duration_ind])
        dims = [self.dim_actions, self.mdp.horizon]
        return [to_onehot(ind, dim) for ind, dim in zip(inds, dims)]

    def representation_to_state(self, state_rep):
        activity_ind = state_rep[0]
        time_ind = int(state_rep[1] * self.interval_length)
        is_done = state_rep[2]
        activity_symbol = self.rev_state_map[activity_ind]
        state_key = (activity_symbol, is_done)
        return self.states[state_key][time_ind]

    def representation_to_action(self, act_rep):
        action_ind = from_onehot(act_rep[0])
        time_ind = (from_onehot(act_rep[1]) * self.interval_length)
        action_symbol = self.rev_action_map[action_ind]
        return self.actions[action_symbol][time_ind]


class TimedActivityMDP(MDP):

    def __init__(self, states, actions, reward_function, gamma, config):
        """

        Args:
            states ():
            actions ():
            gamma (float):
        """
        self.horizon = config.irl_params.horizon
        self.interval_length = config.profile_params.interval_length
        self.activity_states = states['activity']
        self.travel_states = states['travel']
        self.activity_actions = actions['activity']
        self.travel_actions = actions['travel']

        all_states = {}
        all_states.update(self.activity_states)
        all_states.update(self.travel_states)
        self.states = all_states

        all_actions = {}
        all_actions.update(self.activity_actions)
        all_actions.update(self.travel_actions)
        self.actions = all_actions

        transition = TimedActivityTransition(self.states, self.actions,
                                             self.horizon)

        super().__init__(reward_function, transition, gamma)

    def find_state(self, symbol, time, is_done):
        return self.states[(symbol, is_done)][time]

    @property
    def S(self):
        return self.transition.state_id_map

    @property
    def A(self):
        return self.transition.action_id_map

    def available_actions(self, state):
        """

        Args:
            state (DurativeState):

        Returns:
            list[DurativeAction]:
        """
        res = []

        for symbol in state.reachable_symbols:
            possible_actions = list(self.actions[symbol].values())
            if state.is_done:
                res.append(possible_actions[0])
            else:
                current_time = state.time_index
                for action in possible_actions:
                    if current_time + action.duration <= self.horizon and \
                            action.duration != 0:
                        res.append(action)
                    elif current_time + action.duration == self.horizon and \
                            action.duration == 0:
                        res.append(action)

        return res
