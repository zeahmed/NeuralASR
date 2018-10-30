import argparse
import os
import pickle
import sys

import numpy as np

from audiosample import AudioSample
from config import Config


class DataSet:
    def __init__(self, filename, config):
        self.filename = filename
        self.config = config
        self.index = 0
        with open(self.filename, 'r') as f:
            self.X = f.readlines()
            self.X = [os.path.join(os.path.dirname(self.filename), x.strip())
                      for x in self.X]

    def augment_mfcc(self, mfcc):
        r = np.random.randint(self.config.rand_shift * -
                              1, self.config.rand_shift)
        mfcc = np.roll(mfcc, r, axis=0)
        if r > 0:
            mfcc = mfcc[r:, :]
        elif r < 0:
            mfcc = mfcc[:r, :]
        return mfcc

    def load_pkl(self, pklfilename):
        with open(pklfilename, 'rb') as input:
            audiosample = pickle.load(input)
        if self.config.rand_shift > 0:
            audiosample.mfcc = self.augment_mfcc(audiosample.mfcc)
        seq_len = np.asarray(audiosample.mfcc.shape[0], dtype=np.int32)
        labels_len = audiosample.labels.shape[0]
        return audiosample.mfcc, audiosample.labels, seq_len, labels_len

    def reset_epoch(self):
        self.index = 0

    def has_more_batches(self):
        return self.index < len(self.X)

    def get_next_batch(self):
        mfccs, labels, seq_lens, labels_lens = self.load_pkl(
            self.X[self.index])
        mfccs = [mfccs]
        labels = [labels]
        seq_lens = [seq_lens]
        labels_lens = [labels_lens]
        self.index += 1
        max_time = mfccs[0].shape[0]
        while self.index % self.config.batch_size > 0 and self.index < len(self.X):
            mfcc, label, seq_len, labels_len = self.load_pkl(
                self.X[self.index])
            if max_time < mfcc.shape[0]:
                max_time = mfcc.shape[0]
            self.index += 1
            mfccs += [mfcc]
            labels += [label]
            seq_lens += [seq_len]
            labels_lens += [labels_len]

        for i in range(len(mfccs)):
            mfccs[i] = np.pad(mfccs[i], ((0, max_time-mfccs[i].shape[0]),
                                         (0, 0)), 'constant', constant_values=(0, 0))
        mfccs = np.asarray(mfccs)
        return np.asarray(mfccs), labels, seq_lens, labels_lens

    def get_feature_shape(self):
        return [self.config.batch_size, None, self.config.feature_size]

    def get_label_shape(self):
        return [self.config.batch_size, None, 1]

    def get_num_of_sample(self):
        return len(self.X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read data from featurized mfcc files.")
    parser.add_argument("config", help="Configuration file.")
    args = parser.parse_args()

    config = Config(args.config)
    config.epochs = 1
    data = DataSet(config.train_input, config)
    i = 0
    while data.has_more_batches():
        mfccs, labels, seq_len, labels_len = data.get_next_batch()
        print(i, ' - ', mfccs.shape, ' - ',
              labels[0], '-', labels_len)
        i += 1
