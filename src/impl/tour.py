from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np

TRIP = "Trip"
LEG = "Leg"
ACTIVITY = "Activity"


class Tour(object):
    def __init__(self, start_time, end_time, primary_activity=None):
        self.start_time = start_time
        self.end_time = end_time
        self.primary_activity = primary_activity


class TourElement(object):
    """
    Base class for journeys and activities, which comprise the `TourElements`
    of a `Person's` `Schedule`

    Attributes:
        kind: `str`. The kind of tour element this is (e.g., "Activity", "Trip")
        start_time: `int`. When this tour element starts (secs since midnight)
        end_time: `int`. When this tour element ends (secs since midnight)
    """

    def __init__(self, ident, kind, start_time=None, end_time=None, next_activity=None, prev_activity=None):
        self.start_time = start_time
        self.end_time = end_time
        self.next_activity = next_activity
        self.prev_activity = prev_activity
        self.ident = ident
        self.kind = kind
        self._duration = None

    @property
    def duration(self):
        if self._duration is None:
            self._duration = self.end_time - self.start_time
        return self._duration

    def set_previous_activity(self, activity):
        self.next_activity = activity

    def set_next_activity(self, activity):
        self.prev_activity = activity

    def __str__(self):
        return self.ident

    def __repr__(self):
        return "{}: {}".format(self.kind, self.ident)


class ActivityEpisode(TourElement):
    """
    Activity that traveler participates in during daily schedule. Has a well-defined location.

    Attributes:
        activity_type: `str`. Type of activity agent is participating in.
        x: `float`. Longitude of activity facility location.
        y: `float`. Latitude of activity facility location

    """

    # TODO: add facilities
    def __init__(self, activity_type, x=None, y=None, start_time=None, end_time=None, link=None, is_last=False):
        TourElement.__init__(self, activity_type, kind=ACTIVITY, start_time=start_time, end_time=end_time)
        self.x = x
        self.y = y
        self.link = link
        self.is_last = is_last


class TripEpisode(TourElement):
    """
    Sequence of legs on trip

    Attributes:
        mode: `str`. travel mode (e.g., car, bus, walk, etc.)
        trav_time: `int`. secs since midnight
    """

    def __init__(self, mode, trav_time, start_time=None, end_time=None, next_activity=None, prev_activity=None):
        TourElement.__init__(self, mode, kind=TRIP, start_time=start_time, end_time=end_time,
                             next_activity=next_activity, prev_activity=prev_activity)
        self.trav_time = trav_time
        self.stages = []

    @property
    def duration(self):
        return self.trav_time


class Segment(TourElement):
    """
    Travel leg between main 'activities in daily schedule.
    """
    def __init__(self, start_time, end_time, mode, trav_time, distance=0., next_activity=None, prev_activity=None):
        super(Segment, self).__init__(mode, kind=LEG, start_time=start_time, end_time=end_time, next_activity=next_activity,
                                      prev_activity=prev_activity)
        self._distance = distance
        self.trav_time = trav_time

    @property
    def distance(self):
        return self._distance


class ActivityTravelPattern(object):
    """
    A `PlanSequence` for a single plan in a plan set consists of an ordered sequence of `TourElements`
    """

    def __init__(self, tour_elems, activity_scoring_function, trip_scoring_function):
        self._tour_elems = tour_elems
        self._score_map = {ACTIVITY: np.nan, TRIP: np.nan}
        self._activity_scoring_function = activity_scoring_function
        self._trip_scoring_function = trip_scoring_function

    def add_leg(self, leg):
        pass

    def add_activity(self, activity):
        pass

    def score(self):
        if len(self._score_map) == 0:
            return np.nan
        else:
            return np.sum(self._score_map.viewvalues())

    def add_score(self, act, score):
        if np.isnan(self._score_map[act]):
            self._score_map[act] = 0
        self._score_map[act] += score

    @property
    def score_map(self):
        return self._score_map

    def compute_plan_utility(self):
        for el in self._tour_elems:
            if isinstance(el, ActivityEpisode):
                self.add_score(el.ident, self._activity_scoring_function(el))
            elif isinstance(el, Segment):
                self.add_score(el.ident, self._trip_scoring_function(el))
        return self._score_map
