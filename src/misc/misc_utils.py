import functools


def lazy_property(fn):
    attr = '_cache_' + fn.__name__

    @property
    @functools.wraps(fn)
    def check_attr(self):
        if not hasattr(self, attr):
            setattr(self, attr, fn)
        return getattr(self, attr)

    return check_attr
