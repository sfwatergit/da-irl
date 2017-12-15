"""
Defines experiment-specific configuration parameters for da-irl.
"""
from src.impl.activity_model import ActivityModel, TravelModel
from src.misc.config import ConfigManager
from src.util.misc_utils import bag_by_type

TRUTHY = ["true", 1, "True", "TRUE", "t", "y", "yes"]


class IRLConfig(ConfigManager):
    def __init__(self, data, json_file=None):
        super(IRLConfig, self).__init__()
        self.num_iters = data.pop('num_iters', 10)
        self.traces_file_path = data.pop('traces_file_path', None)
        self.horizon = data.pop('horizon', 1440)
        self.gamma = data.pop('gamma', 0.999)
        self.avi_tol = data.pop('avi_tol', 1e-4)


class ProfileBuilderConfig(ConfigManager):
    def __init__(self, data, json_file=None):
        super(ProfileBuilderConfig, self).__init__(json_file=json_file)
        self.profile_builder_config_file_path = None


class GeneralConfig(ConfigManager):
    def __init__(self, data, json_file=None):
        super(GeneralConfig, self).__init__(json_file=json_file)
        self.log_path = data.pop("log_path", "data")
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

        self.activity_params = dict((act, ActivityModel(act, atp)) for act, atp in data.pop('activityParams').items())
        self.travel_params = dict((tm, TravelModel(tm, atp)) for tm, atp in data.pop('travelParams').items())

        self.activity_groups = bag_by_type(self.activity_params.values(), lambda x: x.site_type)
        self.home_activity = self.activity_groups['home'][0]
        self.work_activity = self.activity_groups['work'][0]
        self.other_activity = self.activity_groups['other'][0]

        self.maintenance_activity_set = set(map(lambda x: x.symbol,
                                                filter(lambda x: x.is_maintenance, self.activity_params.values())))
