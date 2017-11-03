from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import operator
import os.path
from xml.etree.cElementTree import iterparse

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.impl.tour import ActivityEpisode, Segment

DAY_SECS = 86399
NUM_HRS_AT_END = 24
END_CONSTRAINT = NUM_HRS_AT_END * 60 * 60-1


class Person(object):
    def __init__(cls, pid):
        cls._pid = pid

    def __repr__(self):
        return "ID: {}".format(self._pid)

    @property
    def pid(self):
        return self._pid


class ExpertPerson(Person):
    def __init__(self, pid, plans):
        self.plans = plans
        super(ExpertPerson, self).__init__(pid)


class ExpertTrajectoryData(object):
    def __init__(self, URL, seg_minutes, limit, cache_dir, filter_params, work_act):
        self._filter_params = filter_params
        self._limit = limit
        self._URL = URL
        self._cache_dir = cache_dir
        self._unique_activities = None
        self._tmat = None
        self._seg_minutes = seg_minutes
        self._pop = None
        self._work_act = work_act

    def _get_tmat(self):

        pop_sample = PopulationParser(self._URL, self._cache_dir).parse_population(self._limit)
        self._pop = [ExpertPerson(pid, p) for pid, p in pop_sample.items()]
        seqs = [seq for p in self.filter_pop() for seq in p.plans]
        trajectories = []
        pbar = tqdm(seqs, desc="Segmenting trajectories")
        discarded = 0
        for seq in pbar:
            max_time = max([el.end_time for el in seq])
            min_duration = min([el.duration for el in seq])
            if max_time <= END_CONSTRAINT and min_duration >= 0:
                trajectories.append(segment_sequence(seq, self._seg_minutes))
            else:
                discarded += 1
        print("Discarded {} trajectories for ending past {} hrs".format(discarded, NUM_HRS_AT_END))

        return self._traj_to_mat(trajectories)

    def compute_pop_stats(self):
        dist_dict = {'dur': []}
        ids = []
        for person in self._pop:
            plan = person.plans[0]
            for el in plan:
                if isinstance(el, ActivityEpisode):
                    #                 print el.next_activity.ident
                    if el.ident == self._work_act:
                        ids.append(int(person.pid))
                        # dist_dict['dist'].append(el.distance)
                        dist_dict['dur'].append(el.duration)
        # dist_dict['mode'].append(el.ident)
        return pd.DataFrame(dist_dict, index=ids)

    def filter_pop(self):
        kind, qt = self._filter_params.kind, self._filter_params.qt
        filter_dir = operator.gt if self._filter_params.dir == 'gt' else operator.lt
        df = self.compute_pop_stats()
        cond_list = np.unique(df[kind][filter_dir(df[kind], df[kind].quantile(qt, 'nearest'))].index.values)
        return [person for person in self._pop if int(person.pid) in cond_list]

    def _traj_to_mat(self, trajectories):
        max_len = max([len(a) for a in trajectories])-1
        nT = len(trajectories)
        t_arr =np.zeros([nT,max_len],dtype='S16')
        for t_idx, trajectory in enumerate(trajectories):
            s_len = len(trajectory)
            if s_len <= max_len:
                delta = max_len - s_len
                trajectory.extend([trajectory[-1]] * delta)
            else:
                trajectory = trajectory[:-1]
            t_arr[t_idx] = np.array(trajectory)
        return t_arr.T

    @property
    def tmat(self):
        if self._tmat is None:
            self._tmat = self._get_tmat()
        return self._tmat


class PopulationParser(object):
    def __init__(self, url, cache_dir=None):
        self.url = url
        self._cache_dir = cache_dir

    def parse_population(self, limit):
        population = {}
        i = 0
        pop_cache = self._cache_dir + '/population.pkl.z'
        if not os.path.isfile(pop_cache):
            with tqdm(iterparse(self.url), desc="Parsing population", total=limit, unit='plan') as pbar:
                for step in iterparse(self.url):
                    if i >= limit:
                        break
                    event, elem = step
                    if elem.tag == 'person':
                        person_elem = elem
                        pid = person_elem.attrib.get('id')
                        if pid not in population.keys():
                            population[pid] = self.get_person_data(person_elem, selected_only=False)
                        i += 1
                        pbar.update(1)
            joblib.dump(population, pop_cache)
        else:
            population = joblib.load(pop_cache)
        return population

    def get_person_data(self, person_elem, selected_only=False):
        if selected_only:
            plans = person_elem.findall('plan')
            plans = filter(lambda x: x.attrib['selected'] == 'yes', plans)
        else:
            plans = person_elem.findall("plan")
        plan_set = [self.parse_plan(plan) for plan in plans]
        return plan_set

    def parse_plan(self, plan):
        activities = []
        prev_activity = None
        for tour_element in plan:
            if 'type' in tour_element.attrib:
                start_time = str_to_secs("00:00:00") if tour_element == plan[0] else activities[-1].end_time
                activity = self.parse_activity(tour_element, start_time, tour_element == plan[-1])
                if len(activities) > 0:
                    activity.set_previous_activity(activities[-1])
            elif 'mode' in tour_element.attrib:
                activity = self.parse_leg(tour_element, activities[-1].end_time)
            if prev_activity is not None:
                activity.set_previous_activity(prev_activity)
                prev_activity.set_next_activity(activity)
            prev_activity = activity
            activities.append(activity)

        return activities

    def parse_leg(self, leg, start_time):
        mode = leg.attrib['mode']
        trav_time = str_to_secs(leg.attrib['trav_time'])
        distance = float(leg.getchildren()[0].attrib['distance'])
        end_time = start_time + trav_time
        return Segment(start_time, end_time, mode, trav_time, distance=distance)

    def parse_activity(self, activity, start_time, is_last_activity_of_day=False):
        attrib = activity.attrib
        activity_type = attrib['type']
        x = attrib['x']
        y = attrib['y']
        if activity_type == 'pt interaction':
            end_time = start_time
        elif is_last_activity_of_day:
            if start_time < DAY_SECS:  # before end of 'normal day'
                end_time = DAY_SECS
            else:
                end_time = END_CONSTRAINT  # 32 hr
        else:
            end_time = str_to_secs(activity.attrib['end_time'])

        return ActivityEpisode(activity_type, x, y, start_time, end_time)


def segment_sequence(sequence, seg_minutes):
    """
    Segments a plan sequence into `seg_len` minute intervals.

    Args:
        sequence: a list of tour elements providing the plan sequence for one
                    person.

    Returns:

    """
    seg_len = seg_minutes * 60  # convert to seconds as that's the minimum time resolution.

    trajectory = []
    for activity in sequence:
        duration = activity.duration // seg_len
        trailer = activity.duration % seg_len
        if trailer > 0:
            pass
        if activity.kind == 'pt_interaction':  # pt interaction
            continue
        if duration == 0:
            duration = 1
        trajectory.extend([activity.ident] * duration)
    return trajectory


def str_to_secs(time_str):
    time_vals = map(int, time_str.split(':'))
    return time_vals[0] * 3600 + time_vals[1] * 60 + time_vals[2]
