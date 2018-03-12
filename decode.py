import argparse
import os
import time

import numpy as np
import tensorflow as tf

from common import convert_2_str, load_model
from config import Config
from dataset import DataSet
from logger import get_logger

logger = get_logger()


def decode(dataTest, model_dir):
    logger.info('Batch Dimensions: ' + str(dataTest.get_feature_shape()))
    logger.info('Label Dimensions: ' + str(dataTest.get_label_shape()))

    network = __import__('networks.' + config.network,
                         fromlist=('create_network', 'loss', 'model', 'label_error_rate'))

    tf.set_random_seed(1)
    X, T, Y, O = dataTest.get_batch_op()
    is_training = tf.placeholder(tf.bool)

    logits = network.create_network(
        X, T, dataTest.symbols.counter, is_training)
    loss = network.loss(logits, Y, T)
    model, log_prob = network.model(logits, T)
    mean_ler = network.label_error_rate(model, Y)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    load_model(1, sess, saver, model_dir)

    test_time_sec = 0

    global_step = 0
    metrics = {'test_time_sec': 0, 'avg_loss': 0, 'avg_ler': 0}
    while True:
        global_step += 1
        try:
            t0 = time.time()
            output, valid_loss_val,  valid_mean_ler_value, Original_transcript = sess.run(
                [model, loss, mean_ler, O], feed_dict={is_training: False})
            logger.info('Valid: avg_cost = %.4f' % (valid_loss_val) +
                        ', avg_ler = %.4f' % (valid_mean_ler_value))
            metrics['test_time_sec'] = metrics['test_time_sec'] + \
                (time.time() - t0)
            metrics['avg_loss'] += valid_loss_val
            metrics['avg_ler'] += valid_mean_ler_value
            str_decoded = convert_2_str(output, dataTest.symbols)
            logger.info('Decoded: ' + str_decoded)
            logger.info('Original: ' + Original_transcript[0].decode('utf-8'))
        except tf.errors.OutOfRangeError:
            logger.info("Finished Decoding!!!")
            break

    logger.info('Decoded Time = %.4fs, avg_loss = %.4f, avg_ler = %.4f' % (
        metrics['test_time_sec'], metrics['avg_loss'] / global_step, metrics['avg_ler'] / global_step))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Decode test data using trained model.")
    parser.add_argument("-c", "--config", required=True,
                        help="Configuration file.")
    args = parser.parse_args()
    config = Config(args.config, True)
    sym_file = os.path.join(
        config.model_dir, os.path.basename(config.sym_file))
    dataTest = DataSet(config.test_input, sym_file,
                       config.feature_size,  batch_size=1, epochs=1)
    decode(dataTest, config.model_dir)
