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
    def __init__(self, filename, config):
        self.feature_size = config.feature_size
        self.filename = filename
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.config = config
        with open(self.filename, 'r') as f:
            self.X = f.readlines()
            self.X = [os.path.join(os.path.dirname(self.filename), x.strip())
                      for x in self.X]

    def augment_mfcc(self, mfcc):
        print(mfcc.shape)
        r = np.random.randint(self.config.rand_shift * -
                              1, self.config.rand_shift)
        mfcc = np.roll(mfcc, r, axis=0)
        if r > 0:
            mfcc = mfcc[r:, :]
        elif r < 0:
            mfcc = mfcc[:r, :]
        print(mfcc.shape)
        return mfcc

    def load_pkl(self, pklfilename):
        with open(pklfilename, 'rb') as input:
            audiosample = pickle.load(input)
        if self.config.rand_shift > 0:
            audiosample.mfcc = self.augment_mfcc(audiosample.mfcc)
        seq_len = np.asarray(audiosample.mfcc.shape[0], dtype=np.int32)
        return audiosample.mfcc, audiosample.labels, seq_len, audiosample.transcription

    def get_batch_op(self, perform_shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices(self.X)
        dataset = dataset.map(lambda pklfilename: tuple(tf.py_func(
            self.load_pkl, [pklfilename], [tf.float32, tf.int32, tf.int32, tf.string])))
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=(
            [None, self.feature_size], [None], [], []))
        iterator = dataset.make_one_shot_iterator()
        batch_features, labels, seq_len, original_transcript = iterator.get_next()

        indices, values, dense_shape = tf.py_func(self.sparse_tuple_from, [labels, original_transcript], [
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

    def sparse_tuple_from(self, sequences, transcripts):
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            l = len(transcripts[n])
            indices.extend(zip([n] * l, range(l)))
            values.extend(seq[:l])

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.int32)
        shape = np.asarray([len(sequences), np.asarray(
            indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read data from featurized mfcc files.")
    parser.add_argument("config", help="Configuration file.")
    args = parser.parse_args()

    config = Config(args.config)
    config.epochs = 1
    data = DataSet(config.train_input, config)
    next_batch = data.get_batch_op()
    result = tf.add(next_batch[0], next_batch[0])
    with tf.Session() as session:
        i = 0
        while True:
            try:
                mfcc, seq_len, labels, transcript = session.run(next_batch)
                print(i, ' - ', mfcc.shape, ' - ',
                      labels[0], '-', transcript)
                i += 1
            except tf.errors.OutOfRangeError:
                print("End of dataset")  # ==> "End of dataset"
                break
