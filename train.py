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
#from decode import decode_batch
from neuralnetworks import bilstm_model
from preprocess import SpeechSample


def train_model(dataTrain, model_dir):
    print('Batch Dimensions: ', dataTrain.get_feature_shape())
    print('Label Dimensions: ', dataTrain.get_label_shape())

    tf.set_random_seed(1)
    X, T, Y = dataTrain.get_batch_op()
    is_training = tf.placeholder(tf.bool)

    model, loss = bilstm_model(dataTrain, X, Y, T, is_training)
    adam_opt = tf.train.AdamOptimizer(learning_rate=0.01)#.minimize(loss)
    gradients, variables = zip(*adam_opt.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = adam_opt.apply_gradients(zip(gradients, variables))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    global_step = 0
    load_model(global_step, sess, saver, model_dir)

    train_time_sec = 0
    avg_loss = 0
    while True:
        global_step += 1
        try:
            t0 = time.time()
            _, loss_val = sess.run([optimizer, loss], feed_dict={is_training: True})
            train_time_sec = train_time_sec + (time.time() - t0)
            avg_loss += loss_val
        except tf.errors.OutOfRangeError:
            print("Done Training...")
            break

        if global_step % 10 == 0:
            saver.save(sess, os.path.join(model_dir, 'model'), global_step=global_step)
            print('Step: ', '%04d' % (global_step), 'cost = %.4f' %
                  (avg_loss / 10))
            avg_loss = 0
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
    args = parser.parse_args()

    dataTrain = DataSet(args.input, batch_size=5,epochs=50)
    train_model(dataTrain, args.model_dir)
