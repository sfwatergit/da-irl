from abc import ABCMeta

import six


class TourElementModel(six.with_metaclass(ABCMeta)):
    def __init__(self, symbol, tour_element_data):
        self.symbol = symbol
        self.site_type = tour_element_data.pop('site_type', 'other')

    def __eq__(self, other):
        return self.symbol == other.symbol

    def __repr__(self):
        return '{}: {}'.format(self.symbol, self.site_type)


class ActivityModel(TourElementModel):
    def __init__(self, symbol, tour_element_data):
        super(ActivityModel, self).__init__(symbol, tour_element_data)
        self.opening_time = tour_element_data.pop('opening_time', 'undefined')
        self.latest_start_time = tour_element_data.pop('latest_start_time', 'undefined')
        self.earliest_end_time = tour_element_data.pop('earliest_end_time', 'undefined')
        self.closing_time = tour_element_data.pop('closing_time', 'undefined')
        self.typical_duration = tour_element_data.pop('latest_start_time', 'undefined')
        self.minimal_duration = tour_element_data.pop('minimal_duration', 'undefined')
        self.is_maintenance = tour_element_data.pop('is_maintenance', False)
        self.is_joint = tour_element_data.pop('is_joint', False)


class TravelModel(TourElementModel):
    def __init__(self, symbol, tour_element_data):
        super(TravelModel, self).__init__(symbol, tour_element_data)
        self.constant = tour_element_data.pop('constant', 0.0)
        self.marginal_utility_dist_m = tour_element_data.pop('constant', 0.0)
        self.marginal_utility_dist_hr = tour_element_data.pop('constant', -6.0)
