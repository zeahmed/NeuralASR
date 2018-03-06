import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from common import load_model
from audio_dataset import DataSet
from neuralnetworks import bilstm_model
from preprocess import SpeechSample


def train_model(dataTrain, model_dir, learning_rate, datavalid):
    print('Batch Dimensions: ', dataTrain.get_feature_shape())
    print('Label Dimensions: ', dataTrain.get_label_shape())

    tf.set_random_seed(1)
    is_training = tf.placeholder(tf.bool)

    if datavalid:
        X, T, Y = tf.cond(is_training, lambda: dataTrain.get_batch_op(),
                          lambda: datavalid.get_batch_op())
    else:
        X, T, Y = dataTrain.get_batch_op()

    model, loss, mean_ler = bilstm_model(dataTrain, X, Y, T, is_training)

    adam_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)  # .minimize(loss)
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
            metrics['train_time_sec'] = metrics['train_time_sec'] + (time.time() - t0)
            metrics['avg_loss'] += loss_val
            metrics['avg_ler'] += mean_ler_value
        except tf.errors.OutOfRangeError:
            print("Done Training...")
            break

        if global_step % report_step == 0:
            saver.save(sess, os.path.join(model_dir, 'model'), global_step=global_step)
            print('Step: ', '%04d' % (global_step), ', cost = %.4f' %
                  (metrics['avg_loss'] / report_step), ', LER = %.4f' % (metrics['avg_ler'] / report_step))
            metrics['avg_loss'] = 0
            metrics['avg_ler'] = 0
            if datavalid:
                valid_loss_val,  valid_mean_ler_value = sess.run(
                    [loss, mean_ler], feed_dict={is_training: False})
                print('Valid: cost = %.4f' % (valid_loss_val),
                      ', LER = %.4f' % (valid_mean_ler_value))
            #feed_dict = {is_training: False}
            #str_decoded = decode_batch(sess, model, feed_dict)
            #print('Decoded: ', str_decoded)
            #print('Original: ', original)

    print("Finished training!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read data from featurized mfcc files.")
    parser.add_argument("-i", "--input", required=True,
                        help="List of pickle files containing mfcc")
    parser.add_argument("-m", "--model_dir", required=False, default='.model',
                        help="Directory to save model files.")
    parser.add_argument("-e", "--epochs", required=False, default=50, type=int,
                        help="number of training iterations.")
    parser.add_argument("-b", "--batch_size", required=False, default=20, type=int,
                        help="Batch size for model training.")
    parser.add_argument("-lr", "--learning_rate", required=False, default=0.001, type=float,
                        help="Learning rate for optimizer.")
    parser.add_argument("-v", "--valid_input", required=False,
                        help="List of pickle files containing mfcc from validation set.")
    args = parser.parse_args()

    dataTrain = DataSet(args.input, batch_size=args.batch_size, epochs=args.epochs)
    dataValid = None
    if args.valid_input:
        dataValid = DataSet(args.valid_input, batch_size=1, epochs=None)
    train_model(dataTrain, args.model_dir, args.learning_rate, dataValid)
