from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import inspect
from itertools import tee, izip
from os import path, makedirs

import numpy as np

FNOPTS = dict(allow_input_downcast=True, on_unused_input='ignore')


def to_onehot(ind, dim):
    ret = np.zeros(dim,dtype=int)
    ret[ind] = 1
    return ret


def to_onehot_n(inds, dim):
    ret = np.zeros((len(inds), dim))
    ret[np.arange(len(inds)), inds] = 1
    return ret


def from_onehot(v):
    return np.nonzero(v)[0][0]


def from_onehot_n(v):
    if len(v) == 0:
        return []
    return np.nonzero(v)[1]

def flatten_list(lol):
    return [a for b in lol for a in b]


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


# @memoize
# def softmax(x1, x2):
#     """
#     x1: float.
#     x2: float.
#     -> softmax(x1, x2)
#     """
#
#     max_x = max(x1, x2)
#     min_x = min(x1, x2)
#     return max_x + np.log(1 + np.exp(min_x - max_x))

def normalize(vals):
    """
    normalize to (0, max_val)
    input:
      vals: 1d array
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)


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


def weighted_sample(weights, objects):
    """
    Return a random item from objects, with the weighting defined by weights
    (which must sum to 1).
    """
    # An array of the weights, cumulatively summed.
    cs = np.cumsum(weights)
    # Find the index of the first weight over a random value.
    idx = sum(cs < np.random.rand())
    return objects[min(idx, len(objects) - 1)]


def weighted_sample_n(prob_matrix, items):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])
    k = (s < r.reshape((-1, 1))).sum(axis=1)
    n_items = len(items)
    return items[np.minimum(k, n_items - 1)]