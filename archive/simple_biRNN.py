import os
import sys
import time
from datetime import datetime
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import collections
import random
import librosa
import util
from util import FIRST_INDEX

class DataSet:
    def __init__(self, filename, num_steps=40, batch_size=1, epochs=5000, labCol=[1], sep=','):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_dir = os.getcwd()+'/model'
        self.labCol = labCol
        self.cache = {}
        self.batch_idx = 0
        (self.X, self.Y) = self._load_data(filename, sep)
        
    def shuffle_ifNeeded(self):
        # perm = np.arange(self.X.shape[0])
        # np.random.shuffle(perm)
        # self.X = self.X[perm]
        # self.Y = self.Y[perm]
        self.batch_idx = 0
        
    def get_sample_batch(self):
        idx = self.batch_idx
        X_batch, Y_batch = self.get_next_batch()
        self.batch_idx = idx
        return  X_batch, Y_batch 

    def get_next_batch(self):
        if self.batch_idx < self.X.shape[0]:
            if not self.X[self.batch_idx] in self.cache:
                mfcc, target, seq_len, original = util.convert_inputs_to_ctc_format(self.X[self.batch_idx], 8000, self.Y[self.batch_idx])
                self.cache[self.X[self.batch_idx]] = [mfcc, target, seq_len, original]
            c = self.cache[self.X[self.batch_idx]]
            self.batch_idx += 1
            return c[0], c[1], c[2], c[3]
        return None

    def has_more_batches(self):
        return self.batch_idx < self.X.shape[0]

    def get_feature_shape(self):
        return [self.batch_size, None, 13]
        
    def get_label_shape(self):
        return [self.batch_size, None, 1]

    def _load_data(self, fileName, sep):
        data = pd.read_csv(fileName, header = None, sep=sep)
        data.sort_values(by=2, inplace=True)
        train_X = data.ix[:, 0].values.ravel()
        train_Y = data.ix[:, self.labCol].values.ravel()
        return (train_X, train_Y)

def create_model(dataset, features, labels, seq_len, is_training):

    num_hidden=100
    num_layers=1
    num_classes = ord('z') - ord('a') + 1 + 1 + 1
    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, features, seq_len, dtype=tf.float32)

    shape = tf.shape(features)
    batch_s, max_time_steps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))
    loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits, seq_len))
    
    model, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len) #tf.nn.ctc_greedy_decoder(logits, seq_len)
    return model[0], loss

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
        dataTrain.shuffle_ifNeeded()
        while dataTrain.has_more_batches():
            X_batch, Y_batch, seq_len, _ = dataTrain.get_next_batch()
            t0 = time.time()
            _, loss_val = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch, T:seq_len, is_training: True})
            train_time_sec = train_time_sec + (time.time() - t0)
            avg_loss += loss_val
            total_batch += 1
        if total_batch > 0:
            print('Epoch: ', '%04d' % (epoch+1), 'cost = %.4f' % (avg_loss / total_batch))
            dataTrain.shuffle_ifNeeded()
            while dataTrain.has_more_batches():
                X_batch, Y_batch, seq_len, original = dataTrain.get_next_batch()
                d = sess.run(model, feed_dict={X: X_batch, T:seq_len, is_training: False})
                str_decoded = ''.join([chr(x+FIRST_INDEX) for x in np.asarray(d[1])])
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

def test_model(modelData, dataTest):
    print('Batch Dimensions: ', dataTest.get_feature_shape())
    print('Label Dimensions: ', dataTest.get_label_shape())

    is_training = modelData['IS_TRAINING']
    test_time_sec = 0
    meansq = 0
    total_batch = 0
    perfResultDict = {}
    while dataTest.has_more_batches():
        X_batch, Y_batch, seq_len, original = dataTest.get_next_batch()
        t0 = time.time()
        d = modelData['SESSION'].run(modelData['MODEL'], feed_dict={modelData['X']: X_batch, modelData['T']:seq_len, is_training: False})
        str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
        print('--', str_decoded)
        print('**', original)

    perfResultDict['TRAIN TIME'] = modelData['TRAIN TIME']
    perfResultDict['TEST TIME'] = test_time_sec

    return perfResultDict

def evaluateMultiClassClassifier(targetlabel, predictedlabel):
    scoredf = pd.DataFrame({'Y_test' : targetlabel, 'Y_pred' : predictedlabel, 'corr': targetlabel == predictedlabel})
    macro_acc = scoredf.groupby('Y_test').mean().mean()['corr']
    micro_acc = scoredf.mean()['corr']

    perfResultDict = {
        'Accuracy(micro-avg)': micro_acc,
        'Accuracy(macro-avg)': macro_acc
    }
    return perfResultDict

tool_version = tf.__version__
print('Tensorflow version: {}.'.format(tool_version))

# Start the clock!
ptm  = time.time()

print('Loading data...')
dataTrain = DataSet(r'~/zeahmed_data/SpeechData/VCTK-Corpus/sample_train.scp')
# while dataTrain.has_more_batches():
    # x, y, seq_len, original = dataTrain.get_next_batch()
    # print(y)
    # exit()
dataTest = DataSet(r'~/zeahmed_data/SpeechData/VCTK-Corpus/sample_train.scp')
print('Done!\n')

modelData = train_model(dataTrain)
perfResultDict = test_model(modelData, dataTest)
runtime = time.time() - ptm
perfResultDict['RUN TIME'] = runtime
perfResultDict['TOOL VERSION'] = tool_version


print('----------------------------------------\n')
# print('Accuracy(micro-avg): %f\nAccuracy(macro-avg): %f\nTraining Time:%f(secs)\nTest/Evaluation Time:%f(secs)\nTotal Time:%f(secs)\n' % (
                # perfResultDict['Accuracy(micro-avg)'],
                # perfResultDict['Accuracy(macro-avg)'],
                # perfResultDict['TRAIN TIME'],
                # perfResultDict['TEST TIME'],
                # perfResultDict['RUN TIME'])
                # )
print('----------------------------------------\n')
