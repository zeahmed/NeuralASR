import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from audiosample import AudioSample
from config import Config


class DataSet:
    START_INDEX = ord('a') - 1

    def __init__(self, filename, feature_size, batch_size=1, epochs=50):
        self.feature_size = feature_size
        self.filename = filename
        self.batch_size = batch_size
        self.epochs = epochs
        with open(self.filename, 'r') as f:
            self.X = f.readlines()
            self.X = [ os.path.join(os.path.dirname(self.filename), x.strip()) for x in self.X]

    def load_pkl(self, pklfilename):
        with open(pklfilename, 'rb') as input:
            audiosample = pickle.load(input)
        return audiosample.mfcc, audiosample.seq_len, audiosample.transcription

    def get_batch_op(self, perform_shuffle=False):
        dataset = tf.contrib.data.Dataset.from_tensor_slices(self.X)
        dataset = dataset.map(lambda pklfilename: tuple(tf.py_func(
            self.load_pkl, [pklfilename], [tf.float32, tf.int32, tf.string])))
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=(
            [None, self.feature_size], [], []))
        iterator = dataset.make_one_shot_iterator()
        batch_features, seq_len, original_transcript = iterator.get_next()

        indices, values, dense_shape = tf.py_func(self.sparse_tuple_from, [original_transcript], [
                                                  tf.int64, tf.int32, tf.int64])
        batch_transcript = tf.SparseTensor(
            indices=indices, values=values, dense_shape=dense_shape)
        return batch_features, seq_len, batch_transcript, original_transcript

    def get_feature_shape(self):
        return [self.batch_size, None, self.feature_size]

    def get_label_shape(self):
        return [self.batch_size, None, 1]

    def get_num_of_sample(self):
        return len(self.X)

    def sparse_tuple_from(self, sequences):
        for i, text in enumerate(sequences):
            sequences[i] = np.asarray(
                [0 if x == 32 else x - DataSet.START_INDEX for x in text])

        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.int32)
        shape = np.asarray([len(sequences), np.asarray(
            indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read data from featurized mfcc files.")
    parser.add_argument("-c", "--config", required=True,
                        help="Configuration file.")
    args = parser.parse_args()

    config = Config(args.config)
    data = DataSet(config.train_input, config.feature_size,
                   batch_size=config.batch_size, epochs=config.epochs)
    next_batch = data.get_batch_op()
    result = tf.add(next_batch[0], next_batch[0])
    with tf.Session() as session:
        i = 0
        while True:
            try:
                mfcc, seq_len, labels, transcript = session.run(next_batch)
                print(i, mfcc.shape, labels, transcript)
                i += 1
                break
            except tf.errors.OutOfRangeError:
                print("End of dataset")  # ==> "End of dataset"
                break
