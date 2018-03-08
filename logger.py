import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def get_logger():
    return logging.getLogger('custom_Speech2Text')
