import os
import random
from itertools import groupby

import numpy as np


def set_global_seeds(i):
    """
    Set the global seed for tensorflow, numpy, and random.

    Args:
        i (int): the seed
    """
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def get_trace_fnames(trace_dir, n=2):
    print('Looking for traces')
    import re
    itr_reg = re.compile(r"traces_persona_(?P<expert_num>[0-9]+)\.csv")

    expert_file_data = []
    for idx, trace_file in enumerate(os.listdir(trace_dir)):
        m = itr_reg.match(trace_file)
        if m:
            expert_num = m.group('expert_num')
            expert_file_data.append([expert_num, os.path.join(trace_dir, m.group())])

    expert_file_data = sorted(expert_file_data, key=lambda x: int(x[0]), reverse=False)[:n]
    for fname in expert_file_data:
        yield fname[1]


def get_expert_fnames(log_dir, n=5):
    """
    Get the filenames for the expert agents in a given log directory.
    These should each contain a dataset for the expert by the name of 'params.pkl'.

    Args:
        log_dir: directory used for logging
        n: limit on number of directories to return.
    """
    print('Looking for paths')
    import re
    itr_reg = re.compile(r"expert_(?P<expert_count>[0-9]+)")

    expert_file_data = []
    for i, log_root in enumerate(os.listdir(log_dir)):
        m = itr_reg.match(log_root)
        if m:
            expert_count = m.group('expert_count')
            expert_path = os.path.join(log_dir, m.group())
            expert_files = os.listdir(expert_path)
            if 'params.pkl' in expert_files:
                expert_filename = os.path.join(expert_path, 'params.pkl')
                expert_file_data.append((expert_count, expert_filename))

    expert_file_data = sorted(expert_file_data, key=lambda x: int(x[0]), reverse=True)[:n]
    for fname in expert_file_data:
        yield fname[1]


def sampling_rollout(env, policy, max_path_length):
    path_length = 1
    path_return = 0
    samples = []
    observation = env.reset()
    terminal = False
    while not terminal and path_length <= max_path_length:
        action, _ = policy.get_action(observation)
        next_observation, reward, terminal, _ = env.step(action)
        samples.append((observation, action, reward, terminal, path_length == 1, path_length))
        observation = next_observation
        path_length += 1

    return samples


def bag_by_type(data, keyfunc):
    groups = []
    uniquekeys = []
    data = sorted(data, key=keyfunc)
    for k, g in groupby(data, keyfunc):
        groups.append(list(g))  # Store group iterator as a list
        uniquekeys.append(k)
    return dict((k, v) for k, v in zip(uniquekeys, groups))


class lazy_property(object):
    """
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    .. Note: https://stackoverflow.com/questions/3012421/python-memoising-deferred-lookup-property-decorator/6849299#6849299
    """
    def __init__(self, fget):

        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value


def make_time_string(tidx, segment_minutes):
    """
    Convert minutes since mignight to hrs.
    :return: Time in HH:MM notation
    """
    mm = tidx * segment_minutes
    mm_str = str(mm % 60).zfill(2)
    hh_str = str(mm // 60).zfill(2)
    return "{}:{}".format(hh_str, mm_str)


def str_to_mins(time_str):
    if time_str == 'undefined':
        return -1  # default value
    time_vals = map(int, time_str.split(':'))
    return (time_vals[0] * 3600 + time_vals[1] * 60 + time_vals[2] * 60) / 60