import argparse
import time

import numpy as np
import tensorflow as tf
from utils import compute_mfcc_and_read_transcription

from common import convert_2_str, load_model
from config import Config
from networks import bilstm_model
from symbols import Symbols


def decode(model_dir, mfcc, sym, seq_len):
    X = tf.placeholder(tf.float32, [1, None, mfcc.shape[2]])
    Y = tf.sparse_placeholder(tf.int32)
    T = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool)

    model, loss, mean_ler = bilstm_model(X, Y, T, sym.counter, is_training)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    load_model(1, sess, saver, model_dir)
    output = sess.run(model, feed_dict={
                      X: mfcc, T: seq_len, is_training: False})
    str_decoded = convert_2_str(output, sym)
    print('Decoded: ', str_decoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a given audio file into text using trained model.")
    parser.add_argument("-i", "--input", required=True,
                        help="Audio file path")
    parser.add_argument("-c", "--config", required=True,
                        help="Configuration file.")
    args = parser.parse_args()
    config = Config(args.config, True)

    sym = Symbols(config.sym_file)
    mfcc, seq_len = compute_mfcc_and_read_transcription(
        args.input, config.samplerate, config.numcontext, config.numcep)
    mfcc = np.expand_dims(mfcc, axis=0)
    decode(config.model_dir, mfcc, sym, [seq_len])
