import os
import sys

import shutil
import tensorflow as tf


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
