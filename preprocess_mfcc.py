import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from utils import compute_mfcc_and_read_transcription

from audiosample import AudioSample
from config import Config
from symbols import Symbols


def update_symbols(sym, clean_transcription):
    for c in clean_transcription:
        if c == ' ':
            sym.insert_space()
        else:
            sym.insert_sym(c)


def write_data(data, config, scp_file_name):
    data.sort_values(by=2, inplace=True)
    train_X = data.ix[:, 0].values.ravel()
    train_Y = data.ix[:, 1].values.ravel()
    sym = Symbols()
    with open(scp_file_name, 'w') as f:
        for i in range(len(train_X)):
            print(train_X[i])
            if os.path.exists(train_X[i]) and os.path.exists(train_Y[i]):
                mfcc, seq_len, clean_transcription = compute_mfcc_and_read_transcription(
                    train_X[i], config.samplerate, config.numcontext, config.numcep, config.punc_regex, train_Y[i])

                update_symbols(sym, clean_transcription)
                filename = os.path.basename(train_X[i]).replace(".wav", "")
                filepath = os.path.join(config.mfcc_output, filename + ".pkl")
                f.write(filename + ".pkl\n")
                with open(filepath, 'wb') as output:
                    audio = AudioSample(filename,
                                        mfcc,
                                        seq_len,
                                        clean_transcription)
                    pickle.dump(audio, output, pickle.HIGHEST_PROTOCOL)

    sym.insert_blank()
    sym.write(os.path.join(config.mfcc_output, config.sym_file))


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
    write_data(dataTrain, config, train_scp_file)

    test_scp_file = os.path.join(config.mfcc_output, "test.scp")
    write_data(dataTest, config, test_scp_file)
