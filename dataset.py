import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from preprocess import SpeechSample

class DataSet:
    def __init__(self, filename, batch_size=1, epochs=50, sep=","):
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch_idx = 0
        (self.X, self.Y) = self._load_data(filename, sep)
        
    def reset(self):
        self.batch_idx = 0
        
    def peek_batch(self):
        idx = self.batch_idx
        mfcc, target, seq_len, clean_transcription = self.get_next_batch()
        self.batch_idx = idx
        return  mfcc, target, seq_len, clean_transcription

    def get_next_batch(self):
        if self.batch_idx < self.X.shape[0]:
            with open(self.X[self.batch_idx], 'rb') as input:
                speechsample = pickle.load(input)
                seq_len = [speechsample.mfcc.shape[1]]
                clean_transcription = self.Y[self.batch_idx]
                self.batch_idx += 1
                return speechsample.mfcc, speechsample.target, seq_len, clean_transcription
        return None

    def has_more_batches(self):
        return self.batch_idx < self.X.shape[0]

    def get_feature_shape(self):
        return [self.batch_size, None, 13]
        
    def get_label_shape(self):
        return [self.batch_size, None, 1]

    def _load_data(self, fileName, sep):
        data = pd.read_csv(fileName, header = None, sep=sep)
        train_X = data.ix[:, 0].values.ravel()
        train_Y = data.ix[:, 1].values.ravel()
        return (train_X, train_Y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read data from featurized mfcc files.")
    parser.add_argument("-i", "--input", required=True,
                        help="List of pickle files containing mfcc")
    args = parser.parse_args()
    
    data = DataSet(args.input)
    while data.has_more_batches():
        mfcc, targets , seq_len , _ = data.get_next_batch()
        print(targets)