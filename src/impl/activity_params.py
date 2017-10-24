from src.misc.parameters import Parameters


class IRLParams(Parameters):
    def __init__(self, data):
        self.segment_minutes = data.pop('segment_minutes', 15)
        self.plans_file_url = data.pop('plans_file_url', None)
        self.pop_limit = data.pop('pop_limit', 10)
        self.num_iters = data.pop('num_iters', 10)
        self.energy_level_bins = data.pop('energy_level_bins', 3)


class FilterParams(Parameters):
    def __init__(self, data):
        self.kind = data.pop('kind', 'dur')
        assert self.kind in ['dur', 'dist']
        self.qt = data.pop('qt', 0.5)
        assert 0 <= self.qt <= 1
        self.dir = data.pop('dir', 'gt')
        assert self.dir in ['gt', 'lt']


class MATSimParameters(Parameters):
    def __init__(self, data):
        self.general_params = data.pop('generalParams')

        self.irl_params = IRLParams(data.pop('irlParams'))

        self.filter_params = FilterParams(data.pop('filterParams'))

        self.activity_params = {
            'performing': data.pop('performing', 6.0),
            'waiting': data.pop('waiting', 0.0),
            'lateArrival': data.pop('lateArrival', -18.0),
            'earlyDeparture': data.pop('earlyDeparture', 0.0)}

        activity_params = data.pop('activityParams')

        self.activity_params = {}
        for act in activity_params.keys():
            activity_type_params = activity_params.pop(act)
            self.activity_params[act] = {
                'typicalDuration': activity_type_params.pop('typicalDuration', 'undefined'),
                'openingTime': activity_type_params.pop('openingTime', 'undefined'),
                'minimalDuration': activity_type_params.pop('minimalDuration', 'undefined'),
                'earliestEndTime': activity_type_params.pop('earliestEndTime', 'undefined'),
                'latestStartTime': activity_type_params.pop('latestStartTime', 'undefined'),
                'closingTime': activity_type_params.pop('closingTime', 'undefined'),
                'type': activity_type_params.pop('type', 'other')
            }

        self.home_act = self.find_act_type_name('home')
        self.work_act = self.find_act_type_name('work')
        self.shopping_act = self.find_act_type_name('shopping')
        self.other_act = self.find_act_type_name('other')

        self.travel_params = {
            'utilityOfLineSwitch': data.pop('utilityOfLineSwitch', -3.0),
            'waitingPt': data.pop('waitingPt', -3.0)}
        mode_params = data.pop('modeParams')
        self.travel_params = {}
        for mode in mode_params.keys():
            mode_type_params = mode_params.pop(mode)
            self.travel_params[mode] = {
                'constant': mode_type_params.pop('constant', 0.0),
                'marginalUtilityOfDistance_util_m': mode_type_params.pop('marginalUtilityOfDistance_util_m', 0.0),
                'marginalUtilityOfTraveling_util_hr': mode_type_params.pop('marginalUtilityOfTraveling_util_hr', -6.0)}

    def find_act_type_name(self, act_type):
        # type: (str, dict) -> str
        for act_name, params in self.activity_params.items():
            if params['type'] == act_type:
                return act_name
