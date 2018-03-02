import os
import sys
import time
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from preprocess.utils import START_INDEX
from dataset import DataSet
from neuralnetworks import create_model
from preprocess import SpeechSample

def train_model(dataTrain):
    print('Batch Dimensions: ', dataTrain.get_feature_shape())
    print('Label Dimensions: ', dataTrain.get_label_shape())
    
    tf.set_random_seed(1)
    X = tf.placeholder(tf.float32, dataTrain.get_feature_shape())
    Y = tf.sparse_placeholder(tf.int32)
    T = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool)

    model, loss = create_model(dataTrain, X, Y, T, is_training)
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    train_time_sec = 0
    for epoch in range(dataTrain.epochs):
        avg_loss = 0
        total_batch = 0
        dataTrain.reset()
        while dataTrain.has_more_batches():
            X_batch, Y_batch, seq_len, _ = dataTrain.get_next_batch()
            t0 = time.time()
            _, loss_val = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch, T:seq_len, is_training: True})
            train_time_sec = train_time_sec + (time.time() - t0)
            avg_loss += loss_val
            total_batch += 1
        if total_batch > 0:
            print('Epoch: ', '%04d' % (epoch+1), 'cost = %.4f' % (avg_loss / total_batch))
            dataTrain.reset()
            X_batch, Y_batch, seq_len, original = dataTrain.peek_batch()
            d = sess.run(model, feed_dict={X: X_batch, T:seq_len, is_training: False})
            str_decoded = ''.join([chr(x+START_INDEX) for x in np.asarray(d[1])])
            str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
            str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
            print('--', str_decoded)
            print('**', original)


    modelData = {
        'MODEL': model,
        'SESSION': sess,
        'LOSS':loss,
        'X': X,
        'Y': Y,
        'T': T,
        'IS_TRAINING': is_training,
        'TRAIN TIME': train_time_sec
    }

    return modelData

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read data from featurized mfcc files.")
    parser.add_argument("-i", "--input", required=True,
                        help="List of pickle files containing mfcc")
    args = parser.parse_args()
    
    dataTrain = DataSet(args.input)
    train_model(dataTrain)