import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd

from utils import convert_inputs_to_ctc_format
from speechsample import SpeechSample


def write_data(data, samplerate, numcep, output_dir, scp_file_name):
    data.sort_values(by=2, inplace=True)
    train_X = data.ix[:, 0].values.ravel()
    train_Y = data.ix[:, 1].values.ravel()
    with open(scp_file_name, 'w') as f:
        for i in range(len(train_X)):
            print(train_X[i])
            if os.path.exists(train_X[i]) and os.path.exists(train_Y[i]):
                mfcc, seq_len, clean_transcription = convert_inputs_to_ctc_format(
                    train_X[i], samplerate, numcep, train_Y[i])
                filename = os.path.basename(train_X[i]).replace(".wav", "")
                filepath = os.path.join(output_dir, filename + ".pkl")
                f.write(filepath + ",\"" + clean_transcription + "\"\n")
                with open(filepath, 'wb') as output:
                    speechsample = SpeechSample(filename,
                                                mfcc,
                                                seq_len)
                    pickle.dump(speechsample, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert audio files into mfcc for training ASR")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input csv file. \
                        1st column is audio file path and 2nd column is transcription file path. \
                        Third column is size of audio file (needed for sorting).")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Path to directory where .pkl file will be generated.")
    parser.add_argument("-sr", "--samplerate", required=False, default=8000,
                        help="Audio sample rate to use for conversion and loading audio file.")
    parser.add_argument("-n", "--numcep", required=False, default=13,
                        help="Number of cepstral coefficients")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataTest = pd.read_csv(args.input, header=None, sep=',')
    dataTrain = dataTest.sample(frac=0.8, random_state=200)
    dataTest = dataTest.drop(dataTrain.index)

    np.set_printoptions(suppress=True)

    train_scp_file = os.path.join(args.output_dir, "train.scp")
    write_data(dataTrain, args.samplerate, args.numcep, args.output_dir, train_scp_file)

    test_scp_file = os.path.join(args.output_dir, "test.scp")
    write_data(dataTest, args.samplerate, args.numcep, args.output_dir, test_scp_file)
