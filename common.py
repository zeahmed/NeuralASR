import os
import shutil
import sys

import numpy as np
import tensorflow as tf

from symbols import Symbols


def convert_2_str(output, sym):
    return sym.convert_to_str(np.asarray(output[1]))


def load_model(start_epoch, sess, saver, model_dir):
    if start_epoch > 0:
        print("Restoring checkpoint: " + model_dir, file=sys.stderr)
        model_file = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, model_file)
        print("Done Restoring checkpoint: " + model_file, file=sys.stderr)
    else:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
