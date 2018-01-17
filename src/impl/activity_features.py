import re
from abc import abstractmethod

import numpy as np

from src import ATPConfig
from src.core.mdp import FeatureExtractor
from src.impl.activity_mdp import ActivityState, TravelState, ATPAction
from src.impl.activity_model import PersonModel
from src.util.misc_utils import str_to_mins


class TravelFeature(FeatureExtractor):
    def __init__(self, name, size, person_model, interval_length, **kwargs):
        super(TravelFeature, self).__init__(name, size, **kwargs)
        self.person_model = person_model
        self.interval_length = interval_length

    def __call__(self, state, action):
        raise NotImplementedError("Must implement __call__")


class TravelTimeDisutilityFeature(TravelFeature):
    def __init__(self, mode, person_model, interval_length, **kwargs):
        size = 1
        ident = mode
        super(TravelTimeDisutilityFeature, self).__init__(mode, 1, person_model,
                                                          interval_length,
                                                          **kwargs)

    def __call__(self, state, action):
        if isinstance(state, ActivityState):
            return np.array([0])
        mode_params = self.person_model.travel_models[state.symbol]
        if state.symbol != self.ident:  # handle each mode separately
            return np.zeros(1)

        next_state = self.env.mdps[self.person_model.agent_id].T(state,action)
        stays = True
        if isinstance(next_state, ActivityState):  # transitioning to activity
            stays = False
        if stays:
            return np.array([self.interval_length / 60.])
        else:
            return np.array([self.interval_length / 60.])

    def __str__(self):
        return "%s travel time disutility" % self.ident

    def __repr__(self):
        return self.__str__()




# class TravelDistanceDisutilityFeature(TripFeature):
#     def __init__(self, mode, params, **kwargs):
#         size = 1
#         ident = mode
#         super(TravelDistanceDisutilityFeature, self).__init__(mode, 1,
# params, **kwargs)
#
#     def __call__(self, state, action):
#         # type: (TravelState, ATPAction) -> ndarray
#         assert (isinstance(state, TravelState))
#         mode_params = self.params.travel_params[state.mode]
#         next_state = self._env.states[action.succ_idx]
#
#         stays = True
#         if isinstance(next_state, ActivityState):
#             stays = False
#         if stays:
#             return state.interval_length * np.array([np.random.normal(0, 1,
#  self.size)]) * 60
#         else:
#             return state.interval_length * np.array([np.random.normal(0, 1,
#  self.size)]) * 60
#
#
#     def __str__(self):
#         return "%s travel distance disutility" % self.ident
#
#     def __repr__(self):
#         return self.__str__()
#


class ActivityFeature(FeatureExtractor):
    """
    Specialized `FeatureExtractor` for resolving features of `Activity`
    `TourElement`s.
    """

    def __init__(self, name, size, person_model, interval_length, **kwargs):
        super(ActivityFeature, self).__init__(name, size, **kwargs)
        self.interval_length = interval_length
        self.person_model = person_model

    def __call__(self, state, action):
        print(state, action)

# class EarlyDepartureFeature(ActivityFeature):
#     def __init__(self, activity_params, **kwargs):
#         size = 1
#         ident = 'Early departure'
#         super(EarlyDepartureFeature, self).__init__(ident, size,
# activity_params, **kwargs)
#
#     def __call__(self, state, action):
#         if isinstance(state, TravelState):
#             return np.array([0])
#
#         arrival_time, departure_time = state.time_index *
# state.interval_length, (
#             state.time_index + 1) * state.interval_length
#         current_activity = state.activity_type
#
#         stays = True
#         next_state = self._env.next_state(state, action)
#         if isinstance(next_state, TravelState):
#             stays = False
#
#         opening_time = str_to_mins(self.params.activity_params[
# current_activity]['openingTime'])
#         closing_time = str_to_mins(self.params.activity_params[
# current_activity]['closingTime'])
#         activity_end = departure_time
#
#         if 0 <= closing_time < departure_time:
#             activity_end = closing_time
#         if (opening_time >= 0 and closing_time >= 0) \
#                 and (opening_time > departure_time or closing_time <
# arrival_time):
#             activity_end = departure_time
#
#         earliest_end_time = str_to_mins(self.params.activity_params[
# current_activity]['earliestEndTime'])
#         if 0 <= activity_end < earliest_end_time and not stays:
#             hr_leaving_early = (earliest_end_time - activity_end) / 60.
#             return np.array([hr_leaving_early])
#         else:
#             return np.array([0])


# class LateDepartureFeature(ActivityFeature):
#     def __init__(self, params, **kwargs):
#         size = 1
#         ident = 'Late departure'
#         super(LateDepartureFeature, self).__init__(ident, size, params,
# **kwargs)
#
#     def __call__(self, state, action):
#         if isinstance(state, TravelState):
#             return np.array([0])
#         arrival_time, departure_time = state.time_index *
# state.interval_length, (
#             state.time_index + 1) * state.interval_length
#         current_activity = state.activity_type
#
#         opening_time = str_to_mins(self.params.activity_params[
# current_activity]['openingTime'])
#         closing_time = str_to_mins(self.params.activity_params[
# current_activity]['closingTime'])
#         activity_end = departure_time
#
#         if opening_time >= 0 and arrival_time < opening_time:
#             if 0 <= closing_time < departure_time:
#                 activity_end = closing_time
#         if (opening_time >= 0 and closing_time >= 0) \
#                 and (opening_time > departure_time or closing_time <
# arrival_time):
#             activity_end = departure_time
#
#         if activity_end < departure_time:
#             hr_waiting_after_end = (departure_time - activity_end) / 60.
#             return np.array([hr_waiting_after_end])
#         else:
#             return np.array([0])

# class EndDayAtHomeFeature(ActivityFeature):
#     def __init__(self, person_model, interval_length, **kwargs):
#         size = 1
#         ident = 'End day at home'
#         super(EndDayAtHomeFeature, self).__init__(ident, size, person_model,
#                                                   interval_length,
#                                                   **kwargs)
#
#     def __call__(self, state, action):
#         if state in self.env.home_goal_states:
#             return np.array([1])
#         else:
#             return np.array([0])
#

class EarlyArrivalFeature(ActivityFeature):
    def __init__(self, person_model, interval_length, **kwargs):
        size = 1
        ident = 'Early arrival'
        super(EarlyArrivalFeature, self).__init__(ident, size, person_model,
                                                  interval_length,
                                                  **kwargs)

    def __call__(self, state, action):
        # type: (ActivityState, ATPAction) -> ndarray
        arrival_time, departure_time = state.time_index * \
                                       self.interval_length, (
                                               state.time_index + 1) * \
                                       self.interval_length
        next_state = self.env.mdps[self.person_model.agent_id].T(state,
                                                                 action)[0][1]
        if isinstance(state, ActivityState):
            return np.array([0])
        if isinstance(state, TravelState):
            if not isinstance(next_state, ActivityState):
                return np.array([0])

        arrival_time = next_state.time_index * self.interval_length
        next_activity = next_state.symbol
        opening_time = str_to_mins(
            self.person_model.activity_models[next_activity].latest_start_time)

        if 0 <= arrival_time < opening_time:  # arrived `interval_length`
            # before open.
            hr_arrived_before_start = (opening_time - arrival_time) /60.
            return np.array([hr_arrived_before_start])
        else:
            return np.array([0])




class LateArrivalFeature(ActivityFeature):
    def __init__(self, person_model, interval_length, **kwargs):
        size = 1
        ident = 'Late arrival'
        super(LateArrivalFeature, self).__init__(ident, size, person_model,
                                                 interval_length,
                                                 **kwargs)

    def __call__(self, state, action):
        next_state = self.env.mdps[self.person_model.agent_id].T(state,
                                                                 action)[0][1]
        if isinstance(state, ActivityState):
            return np.array([0])
        if isinstance(state, TravelState):
            if not isinstance(next_state, ActivityState):
                return np.array([0])

        arrival_time = next_state.time_index * self.interval_length
        next_activity = next_state.symbol

        latest_start_time = str_to_mins(
            self.person_model.activity_models[next_activity].latest_start_time)

        if 0 <= latest_start_time < arrival_time:
            hr_late = (arrival_time - latest_start_time) /60.
            return np.array([hr_late])
        else:
            return np.array([0])

# class TooShortDurationFeature(ActivityFeature):
#     def __init__(self, activity_params, **kwargs):
#         size = 1
#         ident = 'Too short duration'
#         super(TooShortDurationFeature, self).__init__(ident, size,
# activity_params, **kwargs)
#
#     def __call__(self, state, action):
#         if isinstance(state, TravelState):
#             return np.array([0])
#         arrival_time, departure_time = state.time_index *
# state.interval_length, (
#             state.time_index + 1) * state.interval_length
#         current_activity = state.activity_type
#
#         stays = True
#         next_state = self._env.states[action.succ_ix]
#         if isinstance(next_state, TravelState) or next_state.activity_type
# != current_activity:
#             stays = False
#
#         opening_time = str_to_mins(self.params.activity_params[
# current_activity]['openingTime'])
#         closing_time = str_to_mins(self.params.activity_params[
# current_activity]['closingTime'])
#         activity_start = arrival_time
#         activity_end = departure_time
#
#         if opening_time >= 0 and arrival_time < opening_time:
#             activity_start = opening_time
#         if 0 <= closing_time < departure_time:
#             activity_end = closing_time
#         if (opening_time >= 0 and closing_time >= 0) \
#                 and (opening_time > departure_time or closing_time <
# arrival_time):
#             activity_start = departure_time
#             activity_end = departure_time
#
#         duration = activity_end - activity_start
#
#         minimal_duration = str_to_mins(self.params.activity_params[
# current_activity]['minimalDuration'])
#
#         if not stays and 0 <= duration < minimal_duration:
#             hr_lt_min_dur = (minimal_duration - duration) / 60.
#             return np.array([hr_lt_min_dur])
#         else:
#             return np.array([0])


def create_act_at_x_features(where, when, interval_length, person_model):
    """

    Args:
        where (int):
        when (str):
        interval_length (int):
        config (ATPConfig):

    Returns:

    """
    name = "{}At{}Feature".format(where, when)

    def __init__(self, **kwargs):
        self.where = where
        self.when = when
        self.start_period = self.when
        self.end_period = self.when + interval_length
        ActivityFeature.__init__(self, re.compile(r"(\w+\d)").findall(
            str(type(self)))[0], 1, person_model, interval_length, **kwargs)

    def __call__(self, state, action):
        if isinstance(state, TravelState):
            return np.array([0])
        arrival_time, departure_time = state.time_index * \
                                       self.interval_length, (
                                               state.time_index + 1) * \
                                       self.interval_length

        if arrival_time < self.start_period or departure_time > self.end_period:
            return np.array([0])

        current_activity = state.symbol
        if current_activity != self.where.symbol:
            return np.array([0])
        activity_end = (state.time_index + 1) * self.interval_length

        opening_time = str_to_mins(
            self.person_model.activity_models[current_activity].opening_time)
        closing_time = str_to_mins(
            self.person_model.activity_models[current_activity].closing_time)

        activity_start = arrival_time

        if opening_time >= 0 and arrival_time < opening_time:
            activity_start = opening_time
        if 0 <= closing_time < departure_time:
            activity_end = closing_time
        if (opening_time >= 0 and closing_time >= 0) \
                and (
                opening_time > departure_time or closing_time < arrival_time):
            activity_start = departure_time
            activity_end = departure_time

        duration = activity_end - activity_start
        if duration <= 0:
            return np.array([0.])
        else:
            util_performing = duration
            return np.array([util_performing])

    return type(name, (ActivityFeature,),
                dict(__init__=__init__, __call__=__call__))