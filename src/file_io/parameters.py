import json
from collections import Mapping, MutableSequence


class Parameters(object):
    def load(self, json_file):
        """ Load parameters from a json file """
        with open(json_file, 'r') as f:
            jdata = json.load(f)
            for k, v in jdata.items():
                self.__dict__[k] = v

    def save(self, filename):
        """ Save the parameters to file """
        with open(filename, 'w') as f:
            json.dump(self._to_dict(), f, indent=4, sort_keys=True)

    def __repr__(self):
        return self._to_dict()

    def __str__(self):
        d = self._to_dict()
        return ''.join('{}: {}\n'.format(k, v) for k, v in d.items())

    def _to_dict(self):
        return self.__dict__


class FrozenJSON:
    def __init__(self, mapping):
        self.__data = dict(mapping)

    def __getattr__(self, name):
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        else:
            return FrozenJSON.build(self.__data[name])

    @classmethod
    def build(cls, obj):
        if isinstance(obj, Mapping):
            return cls(obj)
        elif isinstance(obj, MutableSequence):
            return [cls.build(item) for item in obj]
        else:
            return obj


if __name__ == '__main__':
    JSON = "/Users/sfeygin/current_code/python/research/marl/deepmarl/data/misc/IRL_multimodal_scenario_params.json"

    with open(JSON) as fp:
        data = FrozenJSON(json.load(fp))
        print data
