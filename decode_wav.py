import argparse

import numpy as np

from config import Config
from logger import get_logger
from utils import compute_mfcc_and_read_transcription

logger = get_logger()


def decode(config, mfcc, seq_len):
    network = config.load_network(fortraining=False)

    output = network.decode(mfcc, seq_len)
    str_decoded = config.symbols.convert_to_str(output[1])
    logger.info('Decoded: ' + str_decoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a given audio file into text using trained model.")
    parser.add_argument("config", help="Configuration file.")
    parser.add_argument("input", help="Audio file path")
    args = parser.parse_args()
    config = Config(args.config, True)

    mfcc = compute_mfcc_and_read_transcription(
        args.input, config.samplerate, config.numcontext, config.numcep)
    mfcc = np.expand_dims(mfcc, axis=0)
    seq_len = np.asarray(mfcc.shape[1], dtype=np.int32)
    decode(config, mfcc, [seq_len])
