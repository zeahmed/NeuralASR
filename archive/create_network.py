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

class DataSet:
    def __init__(self, filename, num_steps=10, batch_size=256, epochs=20, labCol=[1], sep=',', token_sep=','):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_dir = os.getcwd()+'/model'
        self.labCol = labCol
        self.batch_idx = 0
        self.batch_queue_x = np.empty([0, 20])
        self.batch_queue_y = np.empty([0, 20])
        self.token_sep = token_sep
        (self.X, self.Y) = self._load_data(filename, sep)
        
    def shuffle_ifNeeded(self):
        perm = np.arange(self.X.shape[0])
        np.random.shuffle(perm)
        self.X = self.X[perm]
        self.Y = self.Y[perm]
        self.batch_idx = 0
        self.batch_queue_x = np.empty([0, 20])
        self.batch_queue_y = np.empty([0, 20])

    def _get_mfcc(self, filename):
        #print(filename)
        # load wave file
        wav, _ = librosa.load(filename, mono=True, sr=16000)
        # get mfcc feature
        mfcc = np.transpose(librosa.feature.mfcc(wav, 16000))
        return mfcc
    
    def get_sample_batch(self):
        idx = self.batch_idx
        X_batch, Y_batch = self.get_next_batch()
        self.batch_idx = idx
        return  X_batch, Y_batch 

    def get_next_batch(self):
        while len(self.batch_queue_x) < self.batch_size and  self.batch_idx < self.X.shape[0] :
            mfcc = self._get_mfcc(self.X[self.batch_idx])
            self.batch_queue_x = np.append(self.batch_queue_x, mfcc, axis=0)
            self.batch_queue_y = np.append(self.batch_queue_y, mfcc, axis=0)
            self.batch_idx = self.batch_idx + 1
        
        if len(self.batch_queue_x) < self.batch_size:
            for i in range(self.batch_size - len(self.batch_queue_x)):
                self.batch_queue_x = np.append(self.batch_queue_x, self.batch_queue_x[-1].reshape((1,20)), axis=0)
                self.batch_queue_y = np.append(self.batch_queue_y, self.batch_queue_y[-1].reshape((1,20)), axis=0)
        
        X_batch = self.batch_queue_x[0:self.batch_size]
        Y_batch = self.batch_queue_y[0:self.batch_size]
        self.batch_queue_x = self.batch_queue_x[self.batch_size:]
        self.batch_queue_y = self.batch_queue_y[self.batch_size:]
        
        min = np.min(X_batch, axis=0)
        max = np.max(X_batch, axis=0)
        max_min =  max - min
        X_batch = (X_batch - min) / max_min
        return X_batch, Y_batch

    def has_more_batches(self):
        return self.batch_idx < self.X.shape[0]

    def get_feature_shape(self):
        return [self.batch_size, 20]
        
    def get_label_shape(self):
        return [self.batch_size, 20]

    def _load_data(self, fileName, sep):
        data = pd.read_csv(fileName, header = None, sep=sep)
        colNames = np.array(list(data))
        features = np.delete(colNames,self.labCol)
        train_X = data.ix[:, features].values.ravel()
        train_Y = data.ix[:, self.labCol].values.ravel()
        return (train_X, train_Y)


def dense(inputs, units, activation, over_time):
    input_shape = inputs.get_shape().as_list()
    n_input = input_shape[-1]
    Wh = tf.Variable(tf.random_uniform((n_input, units), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
    bh = tf.Variable(tf.zeros([units]))
    output = activation(tf.matmul(inputs,Wh) + bh)
    return output, Wh

def densetied(inputs, Wh, activation, over_time):
    units = Wh.get_shape().as_list()[-1]
    print(units)
    bh = tf.Variable(tf.zeros([units]))
    output = activation(tf.matmul(inputs,Wh) + bh)
    return output, Wh

def create_model(dataset, features, labels, is_training):

    layer_1, wh1 = dense(features, 2048, tf.tanh, False)
    layer_2, wh2 = dense(layer_1, 1024, tf.tanh, False)
    layer_3, _ = densetied(layer_2, tf.transpose(wh2), tf.tanh, False)
    layer_4, _ = densetied(layer_3, tf.transpose(wh1), tf.sigmoid, False)
    #layer_4 = layer_4 * max_min + min
    loss = tf.reduce_mean(tf.square(features - layer_4))
    return (layer_4, loss)

def train_model(dataTrain):
    print('Batch Dimensions: ', dataTrain.get_feature_shape())
    print('Label Dimensions: ', dataTrain.get_label_shape())
    
    tf.set_random_seed(1)
    X = tf.placeholder(tf.float32, dataTrain.get_feature_shape())
    Y = tf.placeholder(tf.int64, dataTrain.get_label_shape())
    is_training = tf.placeholder(tf.bool)

    model, loss = create_model(dataTrain, X, Y, is_training)
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    train_time_sec = 0
    for epoch in range(dataTrain.epochs):
        avg_loss = 0
        total_batch = 0
        dataTrain.shuffle_ifNeeded()
        while dataTrain.has_more_batches():
            X_batch, Y_batch = dataTrain.get_next_batch()
            t0 = time.time()
            _, loss_val = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch, is_training: True})
            train_time_sec = train_time_sec + (time.time() - t0)
            avg_loss += loss_val
            total_batch += 1
        if total_batch > 0:
            print('Epoch: ', '%04d' % (epoch+1), 'cost = %.4f' % (avg_loss / total_batch))


    modelData = {
        'MODEL': model,
        'SESSION': sess,
        'LOSS':loss,
        'X': X,
        'Y': Y,
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
        X_batch, Y_batch = dataTest.get_next_batch()
        t0 = time.time()
        mfcc = modelData['SESSION'].run(modelData['MODEL'], feed_dict={modelData['X']: X_batch, is_training: False})
        test_time_sec = test_time_sec + time.time() - t0
        meansq += np.average(np.square(X_batch - mfcc))
        total_batch += 1

    print(np.sqrt(meansq/total_batch))
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
dataTest = DataSet(r'~/zeahmed_data/SpeechData/VCTK-Corpus/sample_train.scp')
print('Done!\n')

modelData = train_model(dataTrain)
perfResultDict = test_model(modelData, dataTest)
runtime = time.time() - ptm
perfResultDict['RUN TIME'] = runtime
perfResultDict['TOOL VERSION'] = tool_version


print('----------------------------------------\n')
print('Accuracy(micro-avg): %f\nAccuracy(macro-avg): %f\nTraining Time:%f(secs)\nTest/Evaluation Time:%f(secs)\nTotal Time:%f(secs)\n' % (
                perfResultDict['Accuracy(micro-avg)'],
                perfResultDict['Accuracy(macro-avg)'],
                perfResultDict['TRAIN TIME'],
                perfResultDict['TEST TIME'],
                perfResultDict['RUN TIME'])
                )
print('----------------------------------------\n')
