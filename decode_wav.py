import argparse
import os
import time

import numpy as np
import tensorflow as tf

from common import convert_2_str, load_model
from config import Config
from logger import get_logger
from symbols import Symbols
from utils import compute_mfcc_and_read_transcription

logger = get_logger()


def decode(model_dir, mfcc, sym, seq_len):
    network = __import__('networks.' + config.network,
                         fromlist=('create_network', 'model'))

    X = tf.placeholder(tf.float32, [1, None, mfcc.shape[2]])
    Y = tf.sparse_placeholder(tf.int32)
    T = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool)

    logits = network.create_network(
        X, T, sym.counter, is_training)
    model, log_prob = network.model(logits, T)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    load_model(1, sess, saver, model_dir)
    output = sess.run(model, feed_dict={
                      X: mfcc, T: seq_len, is_training: False})
    str_decoded = convert_2_str(output, sym)
    logger.info('Decoded: ' + str_decoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a given audio file into text using trained model.")
    parser.add_argument("-i", "--input", required=True,
                        help="Audio file path")
    parser.add_argument("-c", "--config", required=True,
                        help="Configuration file.")
    args = parser.parse_args()
    config = Config(args.config, True)

    sym_file = os.path.join(
        config.model_dir, os.path.basename(config.sym_file))
    sym = Symbols(sym_file)
    mfcc, seq_len = compute_mfcc_and_read_transcription(
        args.input, config.samplerate, config.numcontext, config.numcep)
    mfcc = np.expand_dims(mfcc, axis=0)
    decode(config.model_dir, mfcc, sym, [seq_len])
