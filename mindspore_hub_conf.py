"""MindSpore Hub config"""

from src.network import WRN


def create_network(name, *args, **kwargs):
    """Creates a WideResNet."""
    if name == 'WRN':
        return WRN(*args, **kwargs)
    raise NotImplementedError('%s is not implemented in the repo' % name)
