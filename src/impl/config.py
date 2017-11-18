import json
import logging


class ConfigManager(object):
    """
    Configuration class designed to be subclassed in order to provide extra params or override
    defaults.

    """

    def __init__(self, json_file=None):
        self._json_file = json_file
        if self._json_file is not None:
            self._from_file()

    def _from_file(self):
        """ Load configuration parameters from a json file
        :type json_file: (str) path to json file of configuration parameters
        """

        with open(self._json_file, 'r') as f:
            jdata = json.load(f)
            for k, v in jdata.items():
                self.__dict__[k] = v
        logging.log(level=logging.INFO, msg="Using custom config file in {}: ({})".format(
            self.__class__.__name__, self._json_file, ))

    def save_to_file(self, filename=None):
        """ Save the parameters to file
        :type filename: (str) file path to save json
        """
        if filename is None:
            filename = self._json_file
        try:
            with open(filename, 'w') as f:
                json.dump(self._to_dict(), f, indent=4, sort_keys=True)
        except IOError:
            logging.log(level=logging.ERROR, msg='Must provide a filename if not loading from '
                                                 'config.')

    def _to_dict(self):
        return self.__dict__

    def __repr__(self):
        return self._to_dict().__repr__()

    def __str__(self):
        d = self._to_dict()
        return ''.join('{}: {}\n'.format(k, v) for k, v in d.items())
