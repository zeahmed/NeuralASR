import os
import shutil
import sys

import numpy as np
import tensorflow as tf

from logger import get_logger
from symbols import Symbols

logger = get_logger()


def convert_2_str(output, sym):
    return sym.convert_to_str(np.asarray(output[1]))


def load_model(start_epoch, sess, saver, model_dir):
    if start_epoch > 0:
        logger.info("Restoring checkpoint: " + model_dir)
        model_file = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, model_file)
        logger.info("Done Restoring checkpoint: " + model_file)
    else:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
