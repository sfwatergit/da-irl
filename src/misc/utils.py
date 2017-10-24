import functools
import inspect
from os import path, makedirs

import numpy as np

FNOPTS = dict(allow_input_downcast=True, on_unused_input='ignore')


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