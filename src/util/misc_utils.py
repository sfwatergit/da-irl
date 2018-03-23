from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import multiprocessing
import os
import random
from itertools import groupby

import numpy as np

# Global variable setting number of cpus to use in TF and other algos.
NCPU = multiprocessing.cpu_count()


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


def get_trace_fnames(traces_dir, n=2):
    print('Looking for traces')
    import re
    itr_reg = re.compile(r"traces_persona_(?P<expert_num>[0-9]+)\.csv")

    expert_file_data = []
    for idx, trace_file in enumerate(os.listdir(traces_dir)):
        m = itr_reg.match(trace_file)
        if m:
            expert_num = m.group('expert_num')
            expert_file_data.append(
                [expert_num, os.path.join(traces_dir, m.group())])

    expert_file_data = sorted(expert_file_data, key=lambda x: int(x[0]),
                              reverse=False)[:n]
    for fname in expert_file_data:
        yield fname[1]


def get_expert_fnames(log_dir, n=5):
    """
    Get the filenames for the expert agents in a given log directory.
    These should each contain a dataset for the expert by the name of
    'params.pkl'.

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

    expert_file_data = sorted(expert_file_data, key=lambda x: int(x[0]),
                              reverse=True)[:n]
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
        samples.append((observation, action, reward, terminal, path_length == 1,
                        path_length))
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
    .. Note: https://stackoverflow.com/questions/3012421/python-memoising
    -deferred-lookup-property-decorator/6849299#6849299
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


def make_time_string(tidx, interval_length):
    """
    Convert minutes since mignight to hrs.
    :return: Time in HH:MM notation
    """
    mm = tidx * interval_length
    mm_str = str(mm % 60).zfill(2)
    hh_str = str(mm // 60).zfill(2)
    return "{}:{}".format(hh_str, mm_str)


def str_to_mins(time_str):
    if time_str == 'undefined':
        return -1  # default value
    time_vals = list(map(int, time_str.split(':')))
    return (time_vals[0] * 3600 + time_vals[1] * 60 + time_vals[2]) / 60


def reverse_action_map(actions):
    """Reverses the Action-index->Action mapping.

    Args:
        actions (dict[int,ATPAction]): Action-index->Action mapping.

    Returns:
        (dict[ATPAction,int]): The action-action-index mapping reversed.

    """
    return dict((v.next_state_symbol, k) for k, v in actions.items())

def flatten(arrays):
    return [sub for arr in arrays for sub in arr]

def safezip(*ls):
    assert all(len(l) == len(ls[0]) for l in ls)
    return zip(*ls)

class Trajectory(object):
    """ Encodes relevant information from a single trajectory: states, actions,
    rewards. Use TrajBatch for a batch of these. """

    __slots__ = ('obs_T_Do', 'obsfeat_T_Df', 'adist_T_Pa', 'a_T_Da', 'r_T')
    def __init__(self, obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T):
        assert (
            obs_T_Do.ndim == 2 and obsfeat_T_Df.ndim == 2 and adist_T_Pa.ndim == 2 and a_T_Da.ndim == 2 and r_T.ndim == 1 and
            obs_T_Do.shape[0] == obsfeat_T_Df.shape[0] == adist_T_Pa.shape[0] == a_T_Da.shape[0] == r_T.shape[0]
        )
        self.obs_T_Do = obs_T_Do
        self.obsfeat_T_Df = obsfeat_T_Df
        self.adist_T_Pa = adist_T_Pa
        self.a_T_Da = a_T_Da
        self.r_T = r_T

    def __len__(self):
        return self.obs_T_Do.shape[0]

    # Saving/loading discards obsfeat
    def save_h5(self, grp, **kwargs):
        grp.create_dataset('obs_T_Do', data=self.obs_T_Do, **kwargs)
        grp.create_dataset('adist_T_Pa', data=self.adist_T_Pa, **kwargs)
        grp.create_dataset('a_T_Da', data=self.a_T_Da, **kwargs)
        grp.create_dataset('r_T', data=self.r_T, **kwargs)

    @classmethod
    def LoadH5(cls, grp, obsfeat_fn):
        '''
        obsfeat_fn: used to fill in observation features. if None, the raw observations will be copied over.
        '''
        obs_T_Do = grp['obs_T_Do'][...]
        obsfeat_T_Df = obsfeat_fn(obs_T_Do) if obsfeat_fn is not None else obs_T_Do.copy()
        return cls(obs_T_Do, obsfeat_T_Df, grp['adist_T_Pa'][...], grp['a_T_Da'][...], grp['r_T'][...])


# Utilities for dealing with batches of trajectories with different lengths

def raggedstack(arrays, fill=0., axis=0, raggedaxis=1):
    '''
    Stacks a list of arrays, like np.stack with axis=0.
    Arrays may have different length (along the raggedaxis), and will be padded on the right
    with the given fill value.
    '''
    assert axis == 0 and raggedaxis == 1, 'not implemented'
    arrays = [a[None,...] for a in arrays]
    assert all(a.ndim >= 2 for a in arrays)

    outshape = list(arrays[0].shape)
    outshape[0] = sum(a.shape[0] for a in arrays)
    outshape[1] = max(a.shape[1] for a in arrays) # take max along ragged axes
    outshape = tuple(outshape)

    out = np.full(outshape, fill, dtype=arrays[0].dtype)
    pos = 0
    for a in arrays:
        out[pos:pos+a.shape[0], :a.shape[1], ...] = a
        pos += a.shape[0]
    assert pos == out.shape[0]
    return out


class RaggedArray(object):
    """ Helps us deal with list of arrays of different lengths. """

    def __init__(self, arrays, lengths=None):
        if lengths is None:
            # Without provided lengths, `arrays` is interpreted as a list of arrays
            # and self.lengths is set to the list of lengths for those arrays
            self.arrays = arrays
            self.stacked = np.concatenate(arrays, axis=0)
            self.lengths = np.array([len(a) for a in arrays])
        else:
            # With provided lengths, `arrays` is interpreted as concatenated data
            # and self.lengths is set to the provided lengths.
            self.arrays = np.split(arrays, np.cumsum(lengths)[:-1])
            self.stacked = arrays
            self.lengths = np.asarray(lengths, dtype=int)
        assert all(len(a) == l for a,l in safezip(self.arrays, self.lengths))
        self.boundaries = np.concatenate([[0], np.cumsum(self.lengths)])
        assert self.boundaries[-1] == len(self.stacked)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return self.stacked[self.boundaries[idx]:self.boundaries[idx+1], ...]

    def padded(self, fill=0.):
        return raggedstack(self.arrays, fill=fill, axis=0, raggedaxis=1)


class TrajBatch(object):
    def __init__(self, trajs, obs, obsfeat, adist, a, r, time):
        self.trajs, self.obs, self.obsfeat, self.adist, self.a, self.r, self.time = trajs, obs, obsfeat, adist, a, r, time

    @classmethod
    def FromTrajs(cls, trajs):
        """
        Returning a TrajBatch, despite the method being defined *in* the
        TrajBatch class. Interesting.  The intuition is that it provides an
        alterantive 'constructor' in case we need to make an instance of the
        class with this particular input (only `trajs`). We can get all the
        other information (obs, obsfeat, etc.) just from the trajs, and rather
        than parse it outside the code, we'll do it internally here. Nice!
        """
        assert all(isinstance(traj, Trajectory) for traj in trajs)
        obs = RaggedArray([t.obs_T_Do for t in trajs])
        obsfeat = RaggedArray([t.obsfeat_T_Df for t in trajs])
        adist = RaggedArray([t.adist_T_Pa for t in trajs])
        a = RaggedArray([t.a_T_Da for t in trajs])
        r = RaggedArray([t.r_T for t in trajs])
        time = RaggedArray([np.arange(len(t), dtype=float) for t in trajs])
        return cls(trajs, obs, obsfeat, adist, a, r, time)

    def with_replaced_reward(self, new_r):
        new_trajs = [Trajectory(traj.obs_T_Do, traj.obsfeat_T_Df, traj.adist_T_Pa, traj.a_T_Da, traj_new_r) for traj, traj_new_r in usafezip(self.trajs, new_r)]
        return TrajBatch(new_trajs, self.obs, self.obsfeat, self.adist, self.a, new_r, self.time)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]

    def save_h5(self, f, starting_id=0, **kwargs):
        for i, traj in enumerate(self.trajs):
            traj.save_h5(f.require_group('%06d' % (i+starting_id)), **kwargs)

    @classmethod
    def LoadH5(cls, dset, obsfeat_fn):
        """
        Another way we can create a TrajBatch, if all we have are `dset`, a
        dataset, and `obsfeat_fn`. Amusingly, it actually calls the *other*
        classmethod. Wow.
        """
        return cls.FromTrajs([Trajectory.LoadH5(v, obsfeat_fn) for k, v in dset.iteritems()])
