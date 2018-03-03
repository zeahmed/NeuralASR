import os

import tensorflow as tf


def load_model(start_epoch, saver, model_dir):
    start_epoch = 0
    if start_epoch > 0:
        model_file = os.path.join(model_dir, "model" + str(start_epoch - 1) + ".ckpt")
        if os.path.exists(model_file + ".index"):
            print("Restoring checkpoint: " + model_file, file=sys.stderr)
            saver.restore(sess, model_file)
            print("Done Restoring checkpoint: " + model_file, file=sys.stderr)
        else:
            print(model_file + " does not exists", file=sys.stderr)
            exit(0)
    else:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
