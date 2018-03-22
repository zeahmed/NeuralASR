import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from utils import compute_mfcc_and_read_transcription

from audiosample import AudioSample
from config import Config
from logger import get_logger
from symbols import Symbols

logger = get_logger()


def update_symbols(sym, clean_transcription, n=1):
    labels = []
    num_context = n // 2
    padded_str = ('$' * num_context)
    padded_transcript = padded_str + clean_transcription + padded_str
    for i in range(len(padded_transcript) - num_context):
        id = sym.insert_sym(padded_transcript[i:i + n])
        labels.append(id)
    return np.asarray(labels, dtype=np.int32)


def write_data(data, config, scp_file_name):
    data.sort_values(by=2, inplace=True)
    train_X = data.ix[:, 0].values.ravel()
    train_Y = data.ix[:, 1].values.ravel()
    logger.info('Writing List of MFCC files to: ' + scp_file_name)
    logger.info('Writing MFCC to: ' + config.mfcc_output)
    with open(scp_file_name, 'w') as f:
        for i in range(len(train_X)):
            logger.info(train_X[i])
            if os.path.exists(train_X[i]) and os.path.exists(train_Y[i]):
                mfcc, clean_transcription = compute_mfcc_and_read_transcription(
                    train_X[i], config.samplerate, config.numcontext, config.numcep, config.punc_regex, train_Y[i])

                if len(clean_transcription) <= mfcc.shape[0]:
                    labels = update_symbols(
                        config.symbols, clean_transcription, config.label_context)
                    filename = os.path.basename(train_X[i]).replace(".wav", "")
                    filepath = os.path.join(config.mfcc_output, filename + ".pkl")
                    f.write(filename + ".pkl\n")
                    with open(filepath, 'wb') as output:
                        audio = AudioSample(filename,
                                            mfcc,
                                            labels,
                                            clean_transcription)
                        pickle.dump(audio, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert audio files into mfcc for training ASR")
    parser.add_argument("config", help="Configuration file.")
    args = parser.parse_args()

    config = Config(args.config)

    if not os.path.exists(config.mfcc_output):
        os.makedirs(config.mfcc_output)

    dataTest = pd.read_csv(config.mfcc_input, header=None, sep=',')
    train_rows = int(dataTest.shape[0] * 0.8)
    dataTrain = dataTest.iloc[:train_rows, :]
    dataTest = dataTest.iloc[train_rows:, :]

    np.set_printoptions(suppress=True)

    train_scp_file = os.path.join(config.mfcc_output, "train.scp")
    write_data(dataTrain, config, train_scp_file)

    test_scp_file = os.path.join(config.mfcc_output, "test.scp")
    write_data(dataTest, config, test_scp_file)

    config.symbols.insert_blank()
    config.write_symbols()
