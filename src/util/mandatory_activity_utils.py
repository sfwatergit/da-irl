from itertools import combinations

import numpy as np


def get_mandatory_activities_done(person_model, current_time_index):
    """Compute the possible mandatory activities that have already been
    completed at the __start__ of this timeslice (i.e., prior to the
    current time index).

    The result is a one-hot vector keyed to the set of
    mandatory activities (sorted lexicographically).

    Conditional logic respects the following boundary conditions:
        1. Agent cannot start off with any mandatory activities completed.
        2. It is impossible for an agent to have completed any mandatory
        activities by the time they've reached the second timeslice
        (travel would've been required).

    Args:
        current_time_index (int): The current timeslice index.

    Yields:
        nd.array[str]: A one-hot vector of mandatory activities completed
                      (or an array of all zeros if none have been
                      completed).
    """
    if current_time_index < 2:
        yield np.zeros(len(person_model.mandatory_activity_set), dtype=int)
    else:
        num_possible_ma = min(current_time_index,
                              len(person_model.mandatory_activity_set))
        for i in range(num_possible_ma, -1, -1):
            possible_mad = combinations(person_model.mandatory_activity_set,
                                        i)
            if i == 0:
                yield np.zeros(len(person_model.mandatory_activity_set),
                               dtype=int)
            else:
                for ma in possible_mad:
                    yield np.array(sum([
                        person_model.mandatory_activity_map[a] for a
                        in ma]))


def maybe_increment_mad(person_model, current_mad,
                        next_activity_symbol):
    """Utility function to compute the possible next set of completed
    mandatory activities given a reachable next state symbol.

    Args:
        person_model (PersonModel): PersonModel specifying mandatory
                                    activities.
        current_mad (nd.array[int]): One-hot vector of the current
                                    completed maintenance activities.
        next_activity_symbol (str): Symbol indicative of the next
                                    activity or travel state for which to
                                    compute the next completed
                                    mandatory activities.

    Returns:
        (nd.array[int]): One-hot vector of the projected completed
                         maintenance activities.

    """
    return current_mad.astype(int) + \
           (person_model.mandatory_activity_map[next_activity_symbol]
            .astype(bool) & ~current_mad).astype(int)
