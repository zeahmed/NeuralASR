import argparse
import os
import time

from common import convert_2_str
from config import Config
from dataset import DataSet
from logger import get_logger

logger = get_logger()


def decode(dataTest, config):
    logger.info('Batch Dimensions: ' + str(dataTest.get_feature_shape()))
    logger.info('Label Dimensions: ' + str(dataTest.get_label_shape()))

    network = config.load_network(fortraining=False)

    global_step = 0
    metrics = {'test_time_sec': 0, 'avg_loss': 0, 'avg_ler': 0}
    while dataTest.has_more_batches():
        global_step += 1
        t0 = time.time()
        mfccs, labels, seq_len, Original_transcript = dataTest.get_next_batch()
        output, valid_loss_val,  valid_mean_ler_value = network.evaluate(
            mfccs, labels, seq_len)
        logger.info('Valid: batch_cost = %.4f' % (valid_loss_val) +
                    ', batch_ler = %.4f' % (valid_mean_ler_value))
        metrics['test_time_sec'] = metrics['test_time_sec'] + \
            (time.time() - t0)
        metrics['avg_loss'] += valid_loss_val
        metrics['avg_ler'] += valid_mean_ler_value
        str_decoded = convert_2_str(output, config.symbols)
        logger.info('Decoded: ' + str_decoded)
        logger.info('Original: ' + Original_transcript[0].replace('_', ' '))

    logger.info("Finished Decoding!!!")
    logger.info('Decoded Time = %.4fs, avg_loss = %.4f, avg_ler = %.4f' % (
        metrics['test_time_sec'], metrics['avg_loss'] / global_step, metrics['avg_ler'] / global_step))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Decode test data using trained model.")
    parser.add_argument("config", help="Configuration file.")
    args = parser.parse_args()
    config = Config(args.config, True)
    config.batch_size = 1
    config.epochs = 1
    config.rand_shift = 0
    dataTest = DataSet(config.test_input, config)
    decode(dataTest, config)
