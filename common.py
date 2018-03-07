import os
import shutil
import sys

import numpy as np
import tensorflow as tf


def convert_2_str(output):
    str_decoded = ''.join([chr(x + DataSet.START_INDEX) for x in np.asarray(output[1])])
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
    return str_decoded


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
