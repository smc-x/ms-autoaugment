"""
Package initialization for common utils.
"""

import logging
logger = logging.getLogger('utils')


def init_utils(config):
    """
    Initialize common utils.

    Args:
        config (Config): Contains configurable parameters throughout the
                         project.
    """
    init_logging(config.log_level)
    for k, v in vars(config).items():
        logger.info('%s=%s', k, v)


def init_logging(level=logging.INFO):
    """
    Initialize logging formats.

    Args:
        level (logging level constant): Set to mute logs with lower levels.
    """
    FMT = r'[%(asctime)s][%(name)s][%(levelname)8s] %(message)s'
    DATE_FMT = r'%Y-%m-%d %H:%M:%S'
    logging.basicConfig(format=FMT, datefmt=DATE_FMT, level=level)
