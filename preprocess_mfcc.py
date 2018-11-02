import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

from audiosample import AudioSample
from config import Config
from logger import get_logger
from symbols import Symbols
from utils import compute_mfcc_and_read_transcription

logger = get_logger()


def update_symbols(config, clean_transcription):
    sym = config.symbols
    n = config.label_context
    labels = [sym.get_id(config.start_marker)] if config.start_marker else []
    num_context = n // 2
    padded_str = (config.start_marker * num_context)
    padded_transcript = padded_str + clean_transcription + padded_str
    for i in range(len(padded_transcript) - num_context):
        id = sym.insert_sym(padded_transcript[i:i + n])
        labels.append(id)
    if config.end_marker:
        labels.append(sym.get_id(config.end_marker))
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
            if not os.path.exists(train_X[i]):
                logger.warn(train_X[i] + ' does not exist.')
            elif not os.path.exists(train_Y[i]):
                logger.warn(train_Y[i] + ' does not exist.')
            else:
                mfcc, clean_transcription = compute_mfcc_and_read_transcription(
                    train_X[i], config.samplerate, config.numcontext, config.numcep, config.punc_regex, train_Y[i])

                if len(clean_transcription) <= mfcc.shape[0]:
                    labels = update_symbols(config, clean_transcription)
                    filename = os.path.basename(train_X[i]).replace(".wav", "")
                    filepath = os.path.join(
                        config.mfcc_output, filename + ".pkl")
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

    config.symbols.insert_padding()
    if config.start_marker:
        config.symbols.insert_sym(config.start_marker)
    if config.end_marker:
        config.symbols.insert_sym(config.end_marker)
    train_scp_file = os.path.join(config.mfcc_output, "train.scp")
    write_data(dataTrain, config, train_scp_file)

    test_scp_file = os.path.join(config.mfcc_output, "test.scp")
    write_data(dataTest, config, test_scp_file)

    config.symbols.insert_blank()
    config.write_symbols()
