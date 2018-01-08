from __future__ import print_function

import base64
import csv
import datetime
import inspect
import json
import os
import os.path as osp
import sys
from contextlib import contextmanager
from pickle import loads

import dateutil.tz
import joblib

from src.misc.console import mkdir_p
from src.misc.serializable import Serializable
from src.misc.tabulate import tabulate
from src.util.math_utils import create_dir_if_not_exists

_prefixes = []
_prefix_str = ''

_tabular_prefixes = []
_tabular_prefix_str = ''

_tabular = []

_text_outputs = []
_tabular_outputs = []

_text_fds = {}
_tabular_fds = {}
_tabular_header_written = set()

_snapshot_dir = None
_snapshot_mode = 'all'

_log_tabular_only = False
_header_printed = False


def _add_output(file_name, arr, fds, mode='a'):
    if file_name not in arr:
        mkdir_p(os.path.dirname(file_name))
        arr.append(file_name)
        fds[file_name] = open(file_name, mode)


def _remove_output(file_name, arr, fds):
    if file_name in arr:
        fds[file_name].close()
        del fds[file_name]
        arr.remove(file_name)


def push_prefix(prefix):
    _prefixes.append(prefix)
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def add_text_output(file_name):
    _add_output(file_name, _text_outputs, _text_fds, mode='a')


def remove_text_output(file_name):
    _remove_output(file_name, _text_outputs, _text_fds)


def add_tabular_output(file_name):
    _add_output(file_name, _tabular_outputs, _tabular_fds, mode='wb')


def remove_tabular_output(file_name):
    if _tabular_fds[file_name] in _tabular_header_written:
        _tabular_header_written.remove(_tabular_fds[file_name])
    _remove_output(file_name, _tabular_outputs, _tabular_fds)


def set_snapshot_dir(dir_name):
    global _snapshot_dir
    create_dir_if_not_exists(dir_name)
    _snapshot_dir = dir_name


def get_snapshot_dir():
    return _snapshot_dir


def get_snapshot_mode():
    return _snapshot_mode


def set_snapshot_mode(mode):
    global _snapshot_mode
    _snapshot_mode = mode


def set_log_tabular_only(log_tabular_only):
    global _log_tabular_only
    _log_tabular_only = log_tabular_only


def get_log_tabular_only():
    return _log_tabular_only


def log(s, with_prefix=True, with_timestamp=True):
    out = s
    if with_prefix:
        out = _prefix_str + out
    if with_timestamp:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
        out = "%s | %s" % (timestamp, out)
    if not _log_tabular_only:
        # Also log to stdout
        print(out)
        for fd in _text_fds.values():
            fd.write(out + '\n')
            fd.flush()
        sys.stdout.flush()


def record_tabular(key, val):
    _tabular.append((_tabular_prefix_str + str(key), str(val)))


def push_tabular_prefix(key):
    _tabular_prefixes.append(key)
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


def pop_tabular_prefix():
    del _tabular_prefixes[-1]
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


@contextmanager
def prefix(key):
    push_prefix(key)
    try:
        yield
    finally:
        pop_prefix()


@contextmanager
def tabular_prefix(key):
    push_tabular_prefix(key)
    yield
    pop_tabular_prefix()


class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


table_printer = TerminalTablePrinter()


def dump_tabular(*args, **kwargs):
    if len(_tabular) > 0:
        if _log_tabular_only:
            table_printer.print_tabular(_tabular)
        else:
            for line in tabulate(_tabular).split('\n'):
                log(line, *args, **kwargs)
        tabular_dict = dict(_tabular)
        # Also write to the csv files
        # This assumes that the keys in each iteration won't change!
        for tabular_fd in _tabular_fds.values():
            writer = csv.DictWriter(tabular_fd, fieldnames=tabular_dict.keys())
            if tabular_fd not in _tabular_header_written:
                writer.writeheader()
                _tabular_header_written.add(tabular_fd)
            writer.writerow(tabular_dict)
            tabular_fd.flush()
        del _tabular[:]


def pop_prefix():
    del _prefixes[-1]
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def save_itr_params(itr, params):
    if _snapshot_dir:
        if _snapshot_mode == 'all':
            file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % itr)
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == 'last':
            # override previous params
            file_name = osp.join(_snapshot_dir, 'params.pkl')
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == 'none':
            pass
        else:
            raise NotImplementedError


def log_parameters(log_file, args):
    log_params = {}
    for param_name, param_value in args.__dict__.iteritems():
        log_params[param_name] = param_value
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(log_params, f, indent=2, sort_keys=True)


def stub_to_json(stub_sth):
    if isinstance(stub_sth, StubObject):
        assert len(stub_sth.args) == 0
        data = dict()
        for k, v in stub_sth.kwargs.iteritems():
            data[k] = stub_to_json(v)
        data[
            "_name"] = stub_sth.proxy_class.__module__ + "." + \
                       stub_sth.proxy_class.__name__
        return data
    elif isinstance(stub_sth, StubAttr):
        return dict(
            obj=stub_to_json(stub_sth.obj),
            attr=stub_to_json(stub_sth.attr_name)
        )
    elif isinstance(stub_sth, StubClass):
        return stub_sth.proxy_class.__module__ + "." + \
               stub_sth.proxy_class.__name__
    elif isinstance(stub_sth, dict):
        return {stub_to_json(k): stub_to_json(v) for k, v in
                stub_sth.iteritems()}
    elif isinstance(stub_sth, (list, tuple)):
        return map(stub_to_json, stub_sth)
    elif type(stub_sth) == type(lambda: None):
        return stub_sth.__module__ + "." + stub_sth.__name__
    return stub_sth


def log_parameters_lite(log_file, args):
    log_params = {}
    for param_name, param_value in args.__dict__.iteritems():
        log_params[param_name] = param_value
    if args.args_data is not None:
        stub_method = loads(base64.b64decode(args.args_data))
        method_args = stub_method.kwargs
        log_params["json_args"] = dict()
        for k, v in method_args.items():
            log_params["json_args"][k] = stub_to_json(v)
        kwargs = stub_method.obj.kwargs
        for k in ["baselines", "env", "policy"]:
            if k in kwargs:
                log_params["json_args"][k] = stub_to_json(kwargs.pop(k))
        log_params["json_args"]["algos"] = stub_to_json(stub_method.obj)
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(log_params, f, indent=2, sort_keys=True)


class StubBase(object):
    def __getitem__(self, item):
        return StubMethodCall(self, "__getitem__", args=[item], kwargs=dict())

    def __getattr__(self, item):
        try:
            return super(self.__class__, self).__getattribute__(item)
        except AttributeError:
            if item.startswith("__") and item.endswith("__"):
                raise
            return StubAttr(self, item)

    def __pow__(self, power, modulo=None):
        return StubMethodCall(self, "__pow__", [power, modulo], dict())

    def __call__(self, *args, **kwargs):
        return StubMethodCall(self.obj, self.attr_name, args, kwargs)

    def __add__(self, other):
        return StubMethodCall(self, "__add__", [other], dict())

    def __rmul__(self, other):
        return StubMethodCall(self, "__rmul__", [other], dict())

    def __div__(self, other):
        return StubMethodCall(self, "__div__", [other], dict())

    def __rdiv__(self, other):
        return StubMethodCall(BinaryOp(), "rdiv", [self, other],
                              dict())  # self, "__rdiv__", [other], dict())

    def __rpow__(self, power, modulo=None):
        return StubMethodCall(self, "__rpow__", [power, modulo], dict())


class BinaryOp(Serializable):
    def __init__(self, *args, **kwargs):
        super(BinaryOp, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def rdiv(self, a, b):
        return b / a
        # def __init__(self, opname, a, b):
        #     self.opname = opname
        #     self.a = a
        #     self.b = b


class StubAttr(StubBase):
    def __init__(self, obj, attr_name):
        self.__dict__["_obj"] = obj
        self.__dict__["_attr_name"] = attr_name

    @property
    def obj(self):
        return self.__dict__["_obj"]

    @property
    def attr_name(self):
        return self.__dict__["_attr_name"]

    def __str__(self):
        return "StubAttr(%s, %s)" % (str(self.obj), str(self.attr_name))


class StubMethodCall(StubBase, Serializable):
    def __init__(self, obj, method_name, args, kwargs):
        super(StubMethodCall, self).__init__()
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        self.obj = obj
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return "StubMethodCall(%s, %s, %s, %s)" % (
            str(self.obj), str(self.method_name), str(self.args),
            str(self.kwargs))


class StubObject(StubBase):
    def __init__(self, __proxy_class, *args, **kwargs):
        if len(args) > 0:
            spec = inspect.getargspec(__proxy_class.__init__)
            kwargs = dict(list(zip(spec.args[1:], args)), **kwargs)
            args = tuple()
        self.proxy_class = __proxy_class
        self.args = args
        self.kwargs = kwargs

    def __getstate__(self):
        return dict(args=self.args, kwargs=self.kwargs,
                    proxy_class=self.proxy_class)

    def __setstate__(self, dict):
        self.args = dict["args"]
        self.kwargs = dict["kwargs"]
        self.proxy_class = dict["proxy_class"]

    def __getattr__(self, item):
        # why doesnt the commented code work?
        # return StubAttr(self, item)
        # checks bypassed to allow for accesing instance fileds
        if hasattr(self.proxy_class, item):
            return StubAttr(self, item)
        raise AttributeError(
            'Cannot get attribute %s from %s' % (item, self.proxy_class))

    def __str__(self):
        return "StubObject(%s, *%s, **%s)" % (
            str(self.proxy_class), str(self.args), str(self.kwargs))


class StubClass(StubBase):
    def __init__(self, proxy_class):
        self.proxy_class = proxy_class

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            # Convert the positional arguments to keyword arguments
            spec = inspect.getargspec(self.proxy_class.__init__)
            kwargs = dict(list(zip(spec.args[1:], args)), **kwargs)
            args = tuple()
        return StubObject(self.proxy_class, *args, **kwargs)

    def __getstate__(self):
        return dict(proxy_class=self.proxy_class)

    def __setstate__(self, dict):
        self.proxy_class = dict["proxy_class"]

    def __getattr__(self, item):
        if hasattr(self.proxy_class, item):
            return StubAttr(self, item)
        raise AttributeError

    def __str__(self):
        return "StubClass(%s)" % self.proxy_class
