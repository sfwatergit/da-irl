from abc import ABCMeta

import six


class Person(six.with_metaclass(ABCMeta)):
    def __init__(self, pid, home_label=None, work_label=None, secondary_labels=None):
        """Person representation in da-irl
        Args:
            pid (str): Unique user identifier (required)
            work_label (str): Label for primary work location
            home_label (str): Label for primary home location
            home_label (str): Label for secondary home location
        """
        self._pid = pid
        self._work_label = work_label
        self._home_label = home_label
        self._secondary_labels = secondary_labels

        self._trajectories = None

    def __repr__(self):
        return "ID: {}".format(self._pid)

    @property
    def pid(self):
        return self._pid

    @property
    def home_label(self):
        if self._home_label is None:
            self._home_label = self._get_home_label()
        return self._home_label

    @property
    def work_label(self):
        if self._work_label is None:
            self._work_label = self._get_work_label()
        return self._work_label

    @property
    def secondary_labels(self):
        if self._secondary_labels is None:
            self._secondary_labels = self._get_secondary_labels()
        return self._secondary_labels

    @property
    def trajectories(self):
        """Returns:
            (np.array): trajectories as (N,T) matrix, where T is the length of the trajectory
        """
        if self._trajectories is None:
            self._trajectories = self._compute_trajectories()
        return self._trajectories

    # These methods are expected to be implemented in inheriting classes
    def _compute_trajectories(self):
        raise NotImplementedError('Missing trajectories')

    def _get_home_label(self):
        raise NotImplementedError('Missing home label')

    def _get_work_label(self):
        raise NotImplementedError('Missing work label')

    def _get_secondary_labels(self):
        raise NotImplementedError('Missing secondary labels')




