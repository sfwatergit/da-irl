import os.path as osp

from impl.config import ConfigManager

TRUTHY = ["true", 1, "True", "TRUE", "t", "y", "yes"]


class IRLConfig(ConfigManager):
    def __init__(self, data, json_file=None):
        super(IRLConfig, self).__init__()
        self.num_iters = data.pop('num_iters', 10)
        self.energy_level_bins = data.pop('energy_level_bins', 3)
        self.traces_file_path = data.pop('traces_file_path', None)
        self.horizon = data.pop('horizon', 1440)


class ProfileBuilderConfig(ConfigManager):
    def __init__(self, data, json_file=None):
        super(ProfileBuilderConfig, self).__init__(json_file=json_file)
        self.profile_builder_config_file_path = None
        self.segment_minutes = int('15min'.replace('min', ''))
        self.filter_weekend_days = False
        self.filter_holidays = False


class GeneralConfig(ConfigManager):
    def __init__(self, data, json_file=None):
        super(GeneralConfig, self).__init__(json_file=json_file)
        self.root_path = data.pop("root_path", osp.abspath(osp.join(osp.dirname(__file__), '..')))
        self.log_path = self.root_path + data.pop("log_path", "/data")
        self.profile_builder_config_file_path = data.pop("profile_builder_config_file_path", "/output")
        self.reward_dir = data.pop("reward_dir", "/rewards")
        self.images_dir = data.pop("images_dir", "/images")
        self.run_id = data.pop("run_id", "test_run")


class ATPConfig(ConfigManager):
    def __init__(self, data, json_file=None):
        super(ATPConfig, self).__init__(json_file=json_file)
        self.general_params = GeneralConfig(data.pop('generalParams'))
        self.irl_params = IRLConfig(data.pop('irlParams'))

        self.profile_params = ProfileBuilderConfig(data=None,
                                                   json_file=self.general_params.profile_builder_config_file_path)

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
