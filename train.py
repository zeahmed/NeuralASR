import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from config import Config
from dataset import DataSet
from logger import get_logger

logger = get_logger()

def train_model(dataTrain, datavalid, config):
    logger.info('Batch Dimensions: ' + str(dataTrain.get_feature_shape()))
    logger.info('Label Dimensions: ' + str(dataTrain.get_label_shape()))

    package = config.network.split('.')
    classname = package[-1]
    module = __import__('.'.join(package[:-1]), fromlist=[classname])
    network = getattr(module, classname)(dataTrain, datavalid, config)
    network.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train speech recognizer on featurized mfcc files.")
    parser.add_argument("config", help="Configuration file.")
    args = parser.parse_args()

    config = Config(args.config, isTraining=True)
    dataTrain = DataSet(config.train_input, config)
    dataValid = None
    if config.test_input:
        config_test = Config(args.config)
        config_test.epochs = None
        dataValid = DataSet(config_test.test_input, config_test)
    train_model(dataTrain, dataValid, config)
