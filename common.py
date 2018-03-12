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

def make_parallel(fn, num_gpus, **kwargs):
    """Parallelize given model on multiple gpu devices.
    https://github.com/vahidk/EffectiveTensorflow#make_parallel

    Args:
    fn: Arbitrary function that takes a set of input tensors and outputs a
        single tensor. First dimension of inputs and output tensor are assumed
        to be batch dimension.
    num_gpus: Number of GPU devices.
    **kwargs: Keyword arguments to be passed to the model.
    Returns:
    A tensor corresponding to the model output.
    """
    in_splits = {}
    for k, v in kwargs.items():
        if k in ('num_classes', 'is_training'):
            in_splits[k] = [v] * num_gpus
        elif k in ('Y'):
            in_splits[k] = tf.sparse_split(sp_input=v, num_split=num_gpus, axis=0)
        else:
            in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                outputs = fn(**{k : v[i] for k, v in in_splits.items()})
                for o in range(len(outputs)):
                    if o >= len(out_split):
                        out_split.append([])
                    out_split[o].append(outputs[o])

    
    return [tf.stack(o, axis=0) for o in out_split]