"""
Defines experiment-specific configuration parameters for da-irl.
"""
from src.impl.activity_model import ActivityModel, TravelModel, PersonModel, \
    HouseholdModel
from src.misc.config import ConfigManager


class ATPConfig(ConfigManager):
    def __init__(self, data):
        """Root configuration parameter object.

        This `ConfigManager` object is the parent `ConfigManager` object for
        all of the child classes.

        Args:
            data (dict[str,obj]): Parsed json object mapping root parameter
            keys to values.
        """
        super(ATPConfig, self).__init__()
        self.general_params = GeneralConfig(data.pop('general_params'))
        self.irl_params = IRLConfig(data.pop('irl_params'))
        self.profile_params = ProfileBuilderConfig(
            json_file=self.general_params.profile_builder_config_file_path)
        self.household_params = HouseholdConfig(data.pop('household_data'))


class GeneralConfig(ConfigManager):
    def __init__(self, data):
        """General configuration parameters.

        Args:
            data (dict[str,obj]): Parsed json object mapping `GeneralConfig`
            parameter keys to values.
        """
        super(GeneralConfig, self).__init__()
        self.profile_builder_config_file_path = data.pop(
            'profile_builder_config_file_path')
        self.traces_file_path = data.pop('traces_file_path')
        self.log_path = data.pop("log_path", "data")
        self.reward_dir = data.pop("reward_dir", "/rewards")
        self.images_dir = data.pop("images_dir", "/images")
        self.run_id = data.pop("run_id", "test_run")


class IRLConfig(ConfigManager):
    def __init__(self, data):
        """Configuration parameters specific to the IRL algorithm.

        Args:
            data (dict[str,obj]): Parsed json object mapping `IRLConfig`
            parameter keys to values.
        """
        super(IRLConfig, self).__init__()
        self.num_iters = data.pop('num_iters', 10)
        self.traces_file_path = data.pop('traces_file_path', None)
        self.horizon = data.pop('discretized_horizon', 1440)
        self.gamma = data.pop('gamma', 0.999)
        self.avi_tol = data.pop('avi_tol', 1e-4)


class ProfileBuilderConfig(ConfigManager):
    def __init__(self, json_file):
        """Persona profile builder configuration parameters.

        Args:
            json_file (str): Path to json configuration file (loads persona
            builder configuration parameters from path).
        """
        super(ProfileBuilderConfig, self).__init__(json_file=json_file)
        self.interval_length = int(self.SEQUENCES_RESOLUTION.strip('min'))


class HouseholdConfig(ConfigManager):
    def __init__(self, data):
        """Household configuration parameters.

        Contains one `HouseholdModel` with `PersonModels` for each
        `ExpertPersona`.

        Args:
            data (dict[str,obj]): Parsed household member data.
        """
        super(HouseholdConfig, self).__init__()
        self.household_id = data.pop('household_id')
        member_data = {}
        members = data.pop("members", None)
        for member in members:
            agent_id = member["agent_id"]
            member_data[agent_id] = PersonModel(agent_id,
                dict((act, ActivityModel(act, atp)) for act, atp in
                     member.pop('activity_params').items()),
                dict((tm, TravelModel(tm, atp)) for tm, atp in
                     member.pop('travel_params').items()))

        self.household_model = HouseholdModel(data.pop("household_id", None),
                                              member_data)
