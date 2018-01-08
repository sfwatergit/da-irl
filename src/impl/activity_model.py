from abc import ABCMeta

import six

from src.util.misc_utils import bag_by_type


class TourElementModel(six.with_metaclass(ABCMeta)):
    def __init__(self, symbol, tour_element_data):
        """Basic component of a person's daily activity-travel model.

        Args:
            symbol (str): Used in `da-irl` as an identifier.
            tour_element_data (dict[str,obj]): Parsed json data from the
            master config file.
        """
        self.symbol = symbol
        self.site_type = tour_element_data.pop('site_type', 'other')

    def __eq__(self, other):
        return self.symbol == other.symbol

    def __repr__(self):
        return '{}: {}'.format(self.symbol, self.site_type)


class ActivityModel(TourElementModel):
    def __init__(self, symbol, tour_element_data):
        """Contains information about an activity for a person.

        Args:
            symbol (str): Used as a partial identifier key for this activity
            in the transition graph.
            tour_element_data (dict[str,obj]): Parsed json data from the
            master config file.
        """
        super(ActivityModel, self).__init__(symbol, tour_element_data)
        self.opening_time = tour_element_data.pop('opening_time', 'undefined')
        self.latest_start_time = tour_element_data.pop('latest_start_time',
                                                       'undefined')
        self.earliest_end_time = tour_element_data.pop('earliest_end_time',
                                                       'undefined')
        self.closing_time = tour_element_data.pop('closing_time', 'undefined')
        self.typical_duration = tour_element_data.pop('latest_start_time',
                                                      'undefined')
        self.minimal_duration = tour_element_data.pop('minimal_duration',
                                                      'undefined')
        self.is_mandatory = tour_element_data.pop('is_mandatory', False)
        self.is_joint = tour_element_data.pop('is_joint', False)


class TravelModel(TourElementModel):
    def __init__(self, symbol, tour_element_data):
        """Contains information about a travel mode for a person.

        Args:
            symbol (str): Used as a partial identifier key for this activity
            in the transition graph.
            tour_element_data (dict[str,obj]): Parsed json data from the
            master config file.
        """
        super(TravelModel, self).__init__(symbol, tour_element_data)
        self.constant = tour_element_data.pop('constant', 0.0)
        self.marginal_utility_dist_m = tour_element_data.pop('constant', 0.0)
        self.marginal_utility_dist_hr = tour_element_data.pop('constant', -6.0)


class PersonModel(object):
    def __init__(self, activity_models, travel_models):
        """Contains the information on a single person's schedule/plan.

        To be used for the daily activity-travel utility model transition
        matrix. This includes the activity models
        for each activity and travel models.

        Args:
            activity_models (dict[str,ActivityModel]): Map of activity
            symbols to corresponding ActivityModels.
            travel_models (dict[str,ActivityModel]): Map of travel mode
            symbols to corresponding TravelModels.
        """
        self.travel_models = travel_models
        self.activity_models = activity_models

        self.mandatory_activity_set = set(
            [activity_model.symbol for activity_model in
             activity_models.values()
             if activity_model.is_mandatory])
        self.joint_activity_set = set(
            [activity_model.symbol for activity_model in
             activity_models.values()
             if activity_model.is_joint])

        self.activity_groups = bag_by_type(self.activity_models.values(),
                                           lambda x: x.site_type)

        self.home_activity = self.activity_groups['home'][0]
        self.work_activity = self.activity_groups['work'][0]
        self.other_activity = self.activity_groups['other'][0]


class HouseholdModel(object):
    def __init__(self, household_id, household_member_models):
        # type: (str, dict[str,PersonModel]) -> None
        """Contains the data needed for features and transitions accounting
        for intra-household interactions.

        Note that this model is not data-dependent (the user pre-specifies
        all of the parameters for the experiment).

        Attr:
            home_activity_symbols (list):

        Args:
            household_id (str): Identifier for household
            household_member_models (dict[str,PersonModel]): A dictionary of
            PersonModels representing the members of
            the household. The keys uniquely identify the person.
        """
        self.household_id = household_id
        self.household_member_models = household_member_models

        # Technically unneccessary, but useful in a "Law of Demeter" sense.
        # XXXX: Will be useful when introducing intra-household interactions
        self.home_activity_symbols = [member.home_activity.symbol for member in
                                      household_member_models.values()]
        self.work_activity_symbols = [member.work_activity.symbol for member in
                                      household_member_models.values()]
        self.other_activity_symbols = [member.other_activity.symbol for member
                                       in household_member_models.values()]
