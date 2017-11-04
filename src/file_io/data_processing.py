from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np
from src.file_io.trace_loader import TraceLoader


# BBOX:
LAT_MIN = 32.831
LAT_MAX = 38.67
LNG_MIN = -128.9
LNG_MAX = -117.41

# DOW
MON = 0
TUES = 1
WEDS = 2
THUR = 3
FRI = 4
SAT = 5
SUN = 6


class User:
    def __init__(self, records_path):
        self._records_path = records_path
        self._records = None

        self.name = None
        self.home = None
        self.work = None
        self.weekend_days = [SAT, SUN]  # hard code for now
        self.work_start_hr = 13  # hard code for now
        self.work_end_hr = 16  # hard code for now
        self.early_morning_hr = 5
        self.late_night_hr = 18

        self.attributes = {}
        self._load_records()

    def _load_records(self):
        df = TraceLoader.load_traces_from_csv(self._records_path)

        df = self._process_home_work_other(df)
        df = self._process_speed(df)
        df = self._process_duration(df)
        self._records = df

    def _process_home_work_other(self, df):
        df['at_home'] = 0
        df['at_work'] = 0
        df['at_other'] = 0

        for idx, row in df.iterrows():
            tup = (row.enter_time.dayofweek, row.enter_time.hour, row.exit_time.dayofweek, row.exit_time.hour)
            is_at_home = self._filter_home(tup)
            is_at_work = self._filter_work(tup)
            is_at_other = ((not is_at_home) & (not is_at_work))
            df.loc[idx, 'at_home'] = 1 if is_at_home else 0
            df.loc[idx, 'at_work'] = 1 if is_at_work else 0
            df.loc[idx, 'at_other'] = 1 if is_at_other else 0
        df = df.dropna()
        return df

    def _filter_home(self, tup):
        enter_day, enter_hr, exit_day, exit_hr = tup
        is_same_day = (enter_day == exit_day)
        is_morning_hour = (exit_hr <= self.early_morning_hr)
        is_consecutive_day = (((enter_day == SUN) & (exit_day == MON)) | (enter_day + 1 == exit_day))
        is_late_night = ((enter_hr >= self.late_night_hr) & is_morning_hour)
        cond_1 = (is_consecutive_day & is_late_night)
        cond_2 = (is_same_day & is_morning_hour)
        return cond_1 | cond_2

    def _filter_work(self, tup):
        enter_day, enter_hr, exit_day, exit_hr = tup
        is_work_hour = ((enter_hr > self.work_start_hr) & (exit_hr < self.work_end_hr))
        is_work_day = (enter_day not in self.weekend_days) & (exit_day not in self.weekend_days)
        is_same_day = (enter_day == exit_day)
        return is_work_hour & (is_work_day & is_same_day)

    def _process_speed(self, df):
        cons_pts = np.array(zip(zip(df.lat.values[0:], df.lng.values[0:]), zip(df.lat.values[1:], df.lng.values[1:])))
        pts1 = cons_pts[:, 0]
        pts2 = cons_pts[:, 1]
        dist = np.round(great_circle_distance(pts1, pts2), 5)
        df['dist'] = np.hstack([[0], dist])

        enters = df.enter_time.dt.to_pydatetime()
        exits = df.exit_time.dt.to_pydatetime()
        time_diff = np.diff(zip(exits[0:], enters[1:]))
        time_diff_s = np.apply_along_axis(lambda x: x[0].seconds, 1, time_diff)
        df['time_diff'] = np.vstack([[0], time_diff])
        df['speed'] = np.hstack([[0], np.round(dist / time_diff_s, 5)])
        return df

    def _process_duration(self, df):
        df['duration'] = df.exit_time - df.enter_time
        return df


def great_circle_distance(pt1, pt2):
    """
    Return the great-circle distance in kilometers between arrays of points,
    defined by a np.arrays (lat, lon).
    Examples
    --------
    >>> brussels = np.array([[50.8503, 4.3517]])
    >>> paris = np.array([[48.8566, 2.3522]])
    >>> great_circle_distance(brussels, paris)
    array([ 263.97541641])
    """
    r = 6371.

    delta_latitude = np.radians(pt1[:, 0] - pt2[:, 0])
    delta_longitude = np.radians(pt1[:, 1] - pt2[:, 1])
    latitude1 = np.radians(pt1[:, 0])
    latitude2 = np.radians(pt2[:, 0])

    a = np.sin(delta_latitude / 2) ** 2 + np.cos(latitude1) * np.cos(latitude2) * np.sin(delta_longitude / 2) ** 2
    return r * 2. * np.arcsin(np.sqrt(a))


if __name__ == '__main__':
    # Path
    import sys
    PATH = sys.argv[1]
    u = User(PATH)._records
    print(u)