import numpy as np
from vincenty import vincenty

from core.mdp import FeatureExtractor, TFRewardFunction
from impl.activity_model import PersonModel
from src.core.mdp import RewardFunction
from src.impl.expert_persona import ExpertAgent
from util.misc_utils import str_to_mins


class TimedActivityRewards(TFRewardFunction):

    def __init__(self, features, states, actions, theta=None, rmax=1.0):
        self._states = states
        self._actions = actions
        self._theta = theta
        self._features = features
        self._feature_indices = range(len(self._features))
        #         features = self.activity_features+self.trip_features

        super().__init__(rmax=rmax, features=features)

    def __call__(self, state, action):



    @staticmethod
    def default_features(config, persona_sites):
        return np.concatenate([[TravelDurationDisutilityFeature(config),
                                # TravelDistanceFeature(persona_sites),
                                ActivityDurationFeature('S H', config),
                                LateArrivalFeature('2 W', config),
                                EarlyArrivalFeature('2 W', config),
                                ActivityDurationFeature('2 W', config),
                                ActivityDurationFeature('2 H', config),
                                ActivityDurationFeature('2 o', config),
                                LateArrivalFeature('3 W', config),
                                EarlyArrivalFeature('3 W', config),
                                ActivityDurationFeature('3 W', config),
                                ActivityDurationFeature('3 H', config),
                                ActivityDurationFeature('3 o', config),
                                ActivityDurationFeature('4 W', config),
                                LateArrivalFeature('4 W', config),
                                EarlyArrivalFeature('4 W', config),
                                ActivityDurationFeature('4 H', config),
                                ActivityDurationFeature('4 o', config)
                                ]])

    @property
    def feature_matrix(self):
        dim_S, dim_A = len(self._states), len(self._actions)
        feat_mat = np.zeros((dim_S, dim_A, self.dim_phi))
        for state in self._states.values():
            for action in self._actions.values():
                feat_mat[state.state_id, action.action_id, :] = np.squeeze(
                    self.phi(state, action))
        return feat_mat


class TravelDurationDisutilityFeature(FeatureExtractor):
    def __init__(self, config):
        config = config
        self.interval_length = config.profile_params.interval_length
        household_model = config.household_params.household_model
        self.person_model = \
            list(household_model.household_member_models.values())[0]
        size = 1
        ident = '-'
        super(TravelDurationDisutilityFeature, self).__init__(ident, size)

    def __call__(self, state, action):
        if 'Trip' in state.symbol and "Trip" not in \
                action.next_state_symbol:
            mode_params = self.person_model.travel_models['-']
            duration = action.duration
            return np.array([duration / 60.])
        else:
            return np.array([0.0])

    def __str__(self):
        return "%s travel time disutility" % self.ident

    def __repr__(self):
        return self.__str__()


class TravelDistanceFeature(FeatureExtractor):

    def __init__(self, persona_sites, **kwargs):
        self.persona_sites = persona_sites
        size = 1
        name = 'TravelDistance'
        super(TravelDistanceFeature, self).__init__(name, size, **kwargs)

    def __call__(self, state, action):
        if 'Trip' not in state.symbol and "Trip" in \
                action.next_state_symbol:
            distance = self.compute_travel_distance(state, action)
            return np.array([distance])
        else:
            return np.array([0.0])

    def compute_travel_distance(self, state, action):
        origin_symbol = state.symbol[-1]
        dest_symbol = action.next_state_symbol.split('=> ')[-1][-1]
        origin = self.persona_sites.get_site_for_symbol(origin_symbol).latlng
        dest = self.persona_sites.get_site_for_symbol(dest_symbol).latlng
        distance = vincenty(origin, dest)
        return distance


class ActivityDurationFeature(FeatureExtractor):

    def __init__(self, activity_type, config):
        """

        Args:
            expert (ExpertAgent):
        """

        self.config = config
        self.activity_type = activity_type
        self.interval_length = config.profile_params.interval_length
        # type: int
        self.person_model = \
            list(
                config.household_params.household_model
                    .household_member_models.values())[0]  # type: PersonModel
        super(ActivityDurationFeature, self).__init__("{}: Duration".format(
            self.activity_type), 1)

    def __call__(self, state, action):
        # type: (DurativeState, DurativeAction) -> np.array[float]

        if "=>" in state.symbol:
            return np.array([0])

        arrival_time = state.time_index
        departure_time = arrival_time + action.duration

        activity = state.symbol.partition(' ')[2]

        if state.symbol not in self.activity_type:
            return np.array([0])
        opening_time = str_to_mins(
            self.person_model.activity_models[activity].opening_time)
        closing_time = str_to_mins(
            self.person_model.activity_models[activity].closing_time)

        if arrival_time < opening_time or departure_time > closing_time:
            return np.array([0])

        activity_start = arrival_time
        activity_end = departure_time

        if opening_time >= 0 and arrival_time < opening_time:
            activity_start = opening_time
        if 0 <= closing_time < departure_time:
            activity_end = closing_time
        if (opening_time >= 0 and closing_time >= 0) \
                and (opening_time > departure_time or
                     closing_time < arrival_time):
            activity_start = departure_time
            activity_end = departure_time

        duration = activity_end - activity_start
        if duration <= 0:
            return np.array([0])
        else:
            util_performing = float(duration / 60.)
            return np.array([util_performing])


class LateArrivalFeature(FeatureExtractor):
    def __init__(self, activity_type, config):
        self.activity_type = activity_type
        self.person_model = list(
            config.household_params.household_model
                .household_member_models.values())[0]  # type: PersonModel
        super(LateArrivalFeature, self).__init__("{}: Late Arrival".format(
            self.activity_type), 1)

    def __call__(self, state, action):
        if "=>" in state.symbol:
            return np.array([0])
        if state.is_done:  # departing
            return np.array([0])

        arrival_time = state.time_index

        activity = state.symbol.partition(' ')[2]

        if activity not in self.activity_type:
            return np.array([0])

        latest_start_time = str_to_mins(
            self.person_model.activity_models[activity].latest_start_time)

        if 0 <= latest_start_time < arrival_time:

            hr_late = (arrival_time - latest_start_time) / 60.
            return np.array([hr_late])
        else:
            return np.array([0])


class EarlyArrivalFeature(FeatureExtractor):
    def __init__(self, activity_type, config):
        self.activity_type = activity_type

        self.person_model = list(
            config.household_params.household_model
                .household_member_models.values())[0]  # type: PersonModel
        super(EarlyArrivalFeature, self).__init__("{}: Early Arrival".format(
            self.activity_type), 1)

    def __call__(self, state, action):
        if "=>" in state.symbol:
            return np.array([0])
        if state.is_done:  # departing
            return np.array([0])

        arrival_time = state.time_index

        activity = state.symbol.partition(' ')[2]

        if activity not in self.activity_type:
            return np.array([0])

        opening_time = str_to_mins(
            self.person_model.activity_models[activity].latest_start_time)

        if 0 <= arrival_time < opening_time:

            hr_arrived_before_start = (opening_time - arrival_time) / 60.
            return np.array([hr_arrived_before_start])
        else:
            return np.array([0])
