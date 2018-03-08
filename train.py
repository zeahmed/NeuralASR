import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from common import load_model
from config import Config
from dataset import DataSet
from networks import bilstm_model


def train_model(dataTrain, model_dir, learning_rate, datavalid):
    print('Batch Dimensions: ', dataTrain.get_feature_shape())
    print('Label Dimensions: ', dataTrain.get_label_shape())

    tf.set_random_seed(1)
    is_training = tf.placeholder(tf.bool)

    if datavalid:
        X, T, Y, _ = tf.cond(is_training, lambda: dataTrain.get_batch_op(),
                             lambda: datavalid.get_batch_op())
    else:
        X, T, Y, _ = dataTrain.get_batch_op()

    model, loss, mean_ler = bilstm_model(
        X, Y, T, dataTrain.symbols.counter, is_training)

    adam_opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate)  # .minimize(loss)
    gradients, variables = zip(*adam_opt.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = adam_opt.apply_gradients(zip(gradients, variables))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    global_step = 0
    load_model(global_step, sess, saver, model_dir)

    metrics = {'train_time_sec': 0, 'avg_loss': 0, 'avg_ler': 0}
    report_step = dataTrain.get_num_of_sample() // dataTrain.batch_size
    while True:
        global_step += 1
        try:
            t0 = time.time()
            _, loss_val, mean_ler_value = sess.run(
                [optimizer, loss, mean_ler], feed_dict={is_training: True})
            metrics['train_time_sec'] = metrics['train_time_sec'] + \
                (time.time() - t0)
            metrics['avg_loss'] += loss_val
            metrics['avg_ler'] += mean_ler_value
        except tf.errors.OutOfRangeError:
            print("Done Training...")
            break

        if global_step % report_step == 0:
            saver.save(sess, os.path.join(model_dir, 'model'),
                       global_step=global_step)
            print('Step: ', '%04d' % (global_step), ', cost = %.4f' %
                  (metrics['avg_loss'] / report_step), ', ler = %.4f' % (metrics['avg_ler'] / report_step))
            metrics['avg_loss'] = 0
            metrics['avg_ler'] = 0
            if datavalid:
                valid_loss_val,  valid_mean_ler_value = sess.run(
                    [loss, mean_ler], feed_dict={is_training: False})
                print('Valid: cost = %.4f' % (valid_loss_val),
                      ', ler = %.4f' % (valid_mean_ler_value))

    print("Finished training!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train speech recognizer on featurized mfcc files.")
    parser.add_argument("-c", "--config", required=True,
                        help="Configuration file.")
    args = parser.parse_args()

    config = Config(args.config)
    dataTrain = DataSet(config.train_input, config.sym_file, config.feature_size,
                        batch_size=config.batch_size, epochs=config.epochs)
    dataValid = None
    if config.test_input:
        dataValid = DataSet(config.test_input, config.sym_file, config.feature_size,
                            batch_size=1, epochs=None)

    config.write(os.path.join(config.model_dir, os.path.basename(args.config)))
    dataTrain.symbols.write(os.path.join(config.model_dir, os.path.basename(config.sym_file)))
    train_model(dataTrain, config.model_dir, config.learningrate, dataValid)
