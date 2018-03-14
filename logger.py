import logging
import sys

import info

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_logger():
    return logging.getLogger(info.app_name)
