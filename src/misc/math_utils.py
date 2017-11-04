from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
import inspect
from os import path, makedirs
import math

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


def great_circle_distance(pt1, pt2):
    """
    Return the great-circle distance in kilometers between two points,
    defined by a tuple (lat, lon).
    Examples
    --------
    >>> brussels = (50.8503, 4.3517)
    >>> paris = (48.8566, 2.3522)
    >>> great_circle_distance(brussels, paris)
    263.9754164080347
    """
    r = 6371.

    delta_latitude = math.radians(pt1[0] - pt2[0])
    delta_longitude = math.radians(pt1[1] - pt2[1])
    latitude1 = math.radians(pt1[0])
    latitude2 = math.radians(pt2[0])

    a = math.sin(delta_latitude / 2) ** 2 + math.cos(latitude1) * math.cos(latitude2) * math.sin(delta_longitude / 2) ** 2
    return r * 2. * math.asin(math.sqrt(a))

def compute_distance_matrix(points):
    """
    Return a matrix of distance (in meters) between every point in a given list
    of (lat, lon) location tuples.
    """
    n = len(points)
    return [[1000 * great_circle_distance(points[i], points[j])
             for j in range(n)] for i in range(n)]


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


def softmax(x, t=1):
    '''
    Numerically stable computation of t*log(\sum_j^n exp(x_j / t))

    If the input is a 1D numpy array, computes it's softmax:
        output = t*log(\sum_j^n exp(x_j / t)).
    If the input is a 2D numpy array, computes the softmax of each of the rows:
        output_i = t*log(\sum_j^n exp(x_{ij} / t))

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
    assert t >= 0
    x = np.asarray(x)
    if len(x.shape) == 1: x = x.reshape((1, -1))
    if t == 0: return np.amax(x, axis=1)
    if x.shape[1] == 1: return x

    def softmax_2_arg(x1, x2, t):
        '''
        Numerically stable computation of t*log(exp(x1/t) + exp(x2/t))

        Parameters
        ----------
        x1 : numpy array of shape (n,1)
        x2 : numpy array of shape (n,1)

        Returns
        -------
        numpy array of shape (n,1)
            Each output_i = t*log(exp(x1_i / t) + exp(x2_i / t))
        '''
        tlog = lambda x: t * np.log(x)
        expt = lambda x: np.exp(x / t)

        max_x = np.amax((x1, x2), axis=0)
        min_x = np.amin((x1, x2), axis=0)
        return max_x + tlog(1 + expt((min_x - max_x)))

    sm = softmax_2_arg(x[:, 0], x[:, 1], t)
    # Use the following property of softmax_2_arg:
    # softmax_2_arg(softmax_2_arg(x1,x2),x3) = log(exp(x1) + exp(x2) + exp(x3))
    # which is true since
    # log(exp(log(exp(x1) + exp(x2))) + exp(x3)) = log(exp(x1) + exp(x2) + exp(x3))
    for (i, x_i) in enumerate(x.T):
        if i > 1: sm = softmax_2_arg(sm, x_i, t)
    return sm


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
