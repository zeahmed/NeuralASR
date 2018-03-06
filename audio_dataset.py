import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from preprocess import SpeechSample


class DataSet:
    def __init__(self, filename, batch_size=1, epochs=50, sep=","):
        self.START_INDEX = ord('a') - 1
        self.MFCC_SIZE = 13
        self.filename = filename
        self.batch_size = batch_size
        self.epochs = epochs
        (self.X, self.Y) = self._load_data(filename, sep)

    def load_pkl(self, pklfilename, transcript):
        with open(pklfilename, 'rb') as input:
            speechsample = pickle.load(input)
        return speechsample.mfcc, speechsample.seq_len, transcript

    def get_batch_op(self, perform_shuffle=False):
        dataset = tf.contrib.data.Dataset.from_tensor_slices(
            (self.X, self.Y))
        dataset = dataset.map(lambda pklfilename, transcription: tuple(tf.py_func(
            self.load_pkl, [pklfilename, transcription], [tf.float32, tf.int32, tf.string])))
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=(
            [None, self.MFCC_SIZE], [], []))
        iterator = dataset.make_one_shot_iterator()
        batch_features, seq_len, batch_transcript = iterator.get_next()

        indices, values, dense_shape = tf.py_func(self.sparse_tuple_from, [batch_transcript], [
                                                  tf.int64, tf.int32, tf.int64])
        batch_transcript = tf.SparseTensor(
            indices=indices, values=values, dense_shape=dense_shape)
        return batch_features, seq_len, batch_transcript

    def get_feature_shape(self):
        return [self.batch_size, None, self.MFCC_SIZE]

    def get_label_shape(self):
        return [self.batch_size, None, 1]

    def get_num_of_sample(self):
        return self.X.shape[0]

    def _load_data(self, fileName, sep):
        data = pd.read_csv(fileName, header=None, sep=sep)
        train_X = data.ix[:, 0].values.ravel()
        train_Y = data.ix[:, 1].values.ravel()
        return (train_X, train_Y)

    def sparse_tuple_from(self, sequences):
        for i, text in enumerate(sequences):
            sequences[i] = np.asarray(
                [0 if x == 32 else x - self.START_INDEX for x in text])

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
    parser.add_argument("-i", "--input", required=True,
                        help="List of pickle files containing mfcc")
    parser.add_argument("-e", "--epochs", required=False, default=50, type=int,
                        help="number of training iterations.")
    parser.add_argument("-b", "--batch_size", required=False, default=20, type=int,
                        help="Batch size for model training.")
    args = parser.parse_args()

    data = DataSet(args.input, batch_size=args.batch_size, epochs=args.epochs)
    next_batch = data.get_batch_op()
    result = tf.add(next_batch[0], next_batch[0])
    with tf.Session() as session:
        i = 0
        while True:
            try:
                mfcc, seq_len, labels = session.run(next_batch)
                print(i, mfcc.shape, labels)
                i += 1
                break
            except tf.errors.OutOfRangeError:
                print("End of dataset")  # ==> "End of dataset"
                break
