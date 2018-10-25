import argparse
import os
import sys
import time
from datetime import datetime

from config import Config
from dataset import DataSet
from logger import get_logger

logger = get_logger()

def train_model(dataTrain, datavalid, config):
    logger.info('Batch Dimensions: ' + str(dataTrain.get_feature_shape()))
    logger.info('Label Dimensions: ' + str(dataTrain.get_label_shape()))

    network = config.load_network(fortraining=True)

    metrics = {'train_time_sec': 0, 'avg_loss': 0, 'avg_ler': 0}
    for epoch in range(config.epochs):
        while dataTrain.has_more_batches():
            t0 = time.time()
            mfccs, labels, seq_len, _ = dataTrain.get_next_batch()
            loss, mean_ler = network.train(mfccs, labels, seq_len)
            metrics['train_time_sec'] += (time.time() - t0)
            metrics['avg_loss'] += loss
            metrics['avg_ler'] += mean_ler

            if network.global_step % config.report_step == 0:
                network.save_checkpoint()
                logger.info('Step: %04d' % (network.global_step) + ', cost = %.4f' %
                            (metrics['avg_loss'] / config.report_step) + ', ler = %.4f' % (metrics['avg_ler'] / config.report_step) +
                            ', time = %.4f' % (metrics['train_time_sec']))
                metrics['avg_loss'] = 0
                metrics['avg_ler'] = 0
                if datavalid:
                    if not datavalid.has_more_batches():
                        datavalid.reset_epoch()
                    mfccs, labels, seq_len, _ = datavalid.get_next_batch()
                    valid_loss_val, valid_mean_ler_value = network.validate(mfccs, labels, seq_len)
                    logger.info('Valid: cost = %.4f' % (valid_loss_val) +
                                ', ler = %.4f' % (valid_mean_ler_value))
        dataTrain.reset_epoch()
    
    logger.info("Finished training!!!")


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
