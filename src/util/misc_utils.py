import functools

import os


def lazy_property(fn):
    attr = '_cache_' + fn.__name__

    @property
    @functools.wraps(fn)
    def check_attr(self):
        if not hasattr(self, attr):
            setattr(self, attr, fn)
        return getattr(self, attr)

    return check_attr


def get_expert_fnames(log_dir, n=5):
    """
    Get the filenames for the expert agents in a given log directory.
    These should each contain a dataset for the expert by the name of 'params.pkl'.

    Args:
        log_dir: directory used for logging
        n: limit on number of directories to return.
    """
    print('Looking for paths')
    import re
    itr_reg = re.compile(r"expert_(?P<expert_count>[0-9]+)")

    expert_file_data = []
    for i, log_root in enumerate(os.listdir(log_dir)):
        m = itr_reg.match(log_root)
        if m:
            expert_count = m.group('expert_count')
            expert_path = os.path.join(log_dir, m.group())
            expert_files = os.listdir(expert_path)
            if 'params.pkl' in expert_files:
                expert_filename = os.path.join(expert_path, 'params.pkl')
                expert_file_data.append((expert_count, expert_filename))

    expert_file_data = sorted(expert_file_data, key=lambda x: int(x[0]), reverse=True)[:n]
    for fname in expert_file_data:
        yield fname[1]