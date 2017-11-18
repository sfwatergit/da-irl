from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
import inspect
from itertools import tee, izip
from os import path, makedirs
import math
from cytoolz import memoize

import numpy as np
import functools

FNOPTS = dict(allow_input_downcast=True, on_unused_input='ignore')


def to_onehot(ind, dim):
    ret = np.zeros(dim)
    ret[ind] = 1.0
    return ret


def to_onehot_n(inds, dim):
    ret = np.zeros((len(inds), dim))
    ret[np.arange(len(inds)), inds] = 1
    return ret


def make_time_string(mm):
    # type: (int) -> unicode
    """
    Convert minutes since mignight to hrs.

    :rtype: str
    :param mm: minutes since midnight
    :return: Time in HH:MM notation
    """
    mm_str = str(mm % 60).zfill(2)
    hh_str = str(mm // 60).zfill(2)
    return "{}:{}".format(hh_str, mm_str)

def t2n(hh, mm):
    return hh * 60 + mm


def create_dir_if_not_exists(dir_name):
    # type: (str) -> None
    if not path.exists(dir_name):
        makedirs(dir_name)


def cartesian(lists):
    # type: (list) -> list
    if not lists:
        return [()]
    return [x + (y,) for x in cartesian(lists[:-1]) for y in lists[-1]]


@memoize
def softmax(x1, x2):
    """
    x1: float.
    x2: float.
    -> softmax(x1, x2)
    """

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))

def normalize(vals):
    """
    normalize to (0, max_val)
    input:
      vals: 1d array
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def get_subclass_list(supercls, module=None):
    """
    Introspect the subclasses of a class that are defined within a module
    Args:
        supercls:
        module: referring module. If `None`, then use module where `supercls`' is defined.

    Returns:
            list of subclasses of `supercls` in the module
    """
    if module is None:
        module = inspect.getmodule(inspect.getmro(supercls)[0])
    res = [subcls[1] for subcls in inspect.getmembers(module, inspect.isclass)
           if issubclass(subcls[1], supercls)]
    res.remove(supercls)
    return res


def mellowmax(x, t=1):
    '''
    Numerically stable computation of mellowmax t*log(1/n \sum_j^n exp(x_j/t))

    As per http://proceedings.mlr.press/v70/asadi17a/asadi17a.pdf, this is a
    better version of softmax since mellowmax is a non-expansion an softmax is
    not. The problem is that softmax(1,1,1) is not 1, but instead log(3).
    This causes the softmax value iteration to grow unnecessarily in ie cases
    with no positive reward loops when \gamma=1 and regular value iteration
    would converge.

    If the input is a 1D numpy array, computes it's mellowmax:
        output = t*log(1/n * \sum_j^n exp(x_j / t)).
    If the input is a 2D numpy array, computes the mellowmax of each row:
        output_i = t*log(1/n \sum_j^n exp(x_{ij} / t))

    Parameters
    ----------
    x : 1D or 2D numpy array

    Returns
    -------
    1D numpy array
        shape = (n,), where:
            n = 1 if x was 1D, or
            n is the number of rows (=x.shape[0]) if x was 2D.
    '''
    x = np.asarray(x)
    if len(x.shape) == 1: x = x.reshape((1, -1))
    sm = softmax(x, t=t)
    return sm - t * np.log(x.shape[1])

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    now, nxt = tee(iterable)
    next(nxt, None)
    return izip(now, nxt)

def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 3e-2)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    next_x = None
    beta1, beta2, eps = config['beta1'], config['beta2'], config['epsilon']
    t, m, v = config['t'], config['m'], config['v']
    m = beta1 * m + (1 - beta1) * dx
    v = beta2 * v + (1 - beta2) * (dx * dx)
    t += 1
    alpha = config['learning_rate'] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    x += alpha * (m / (np.sqrt(v) + eps))
    config['t'] = t
    config['m'] = m
    config['v'] = v
    next_x = x

    return next_x, config
