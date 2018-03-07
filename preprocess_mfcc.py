import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from utils import convert_inputs_to_ctc_format

from audiosample import AudioSample
from config import Config


def write_data(data, samplerate, numcontext, numcep, output_dir, scp_file_name):
    data.sort_values(by=2, inplace=True)
    train_X = data.ix[:, 0].values.ravel()
    train_Y = data.ix[:, 1].values.ravel()
    with open(scp_file_name, 'w') as f:
        for i in range(len(train_X)):
            print(train_X[i])
            if os.path.exists(train_X[i]) and os.path.exists(train_Y[i]):
                mfcc, seq_len, clean_transcription = convert_inputs_to_ctc_format(
                    train_X[i], samplerate, numcontext, numcep, train_Y[i])
                filename = os.path.basename(train_X[i]).replace(".wav", "")
                filepath = os.path.join(output_dir, filename + ".pkl")
                f.write(filepath + "\n")
                with open(filepath, 'wb') as output:
                    audio = AudioSample(filename,
                                                mfcc,
                                                seq_len,
                                                clean_transcription)
                    pickle.dump(audio, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert audio files into mfcc for training ASR")
    parser.add_argument("-c", "--config", required=True,
                        help="Configuration file.")
    args = parser.parse_args()

    config = Config(args.config)

    if not os.path.exists(config.mfcc_output):
        os.makedirs(config.mfcc_output)

    dataTest = pd.read_csv(config.mfcc_input, header=None, sep=',')
    dataTrain = dataTest.sample(frac=0.8, random_state=200)
    dataTest = dataTest.drop(dataTrain.index)

    np.set_printoptions(suppress=True)

    train_scp_file = os.path.join(config.mfcc_output, "train.scp")
    write_data(dataTrain, config.samplerate, config.numcontext,
               config.numcep, config.mfcc_output, train_scp_file)

    test_scp_file = os.path.join(config.mfcc_output, "test.scp")
    write_data(dataTest, config.samplerate, config.numcontext,
               config.numcep, config.mfcc_output, test_scp_file)
