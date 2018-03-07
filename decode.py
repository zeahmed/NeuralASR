import argparse
import time

import numpy as np
import tensorflow as tf

from audio_dataset import DataSet
from common import convert_2_str, load_model
from neuralnetworks import bilstm_model
from preprocess import SpeechSample


def decode(dataTest, model_dir):
    print('Batch Dimensions: ', dataTest.get_feature_shape())
    print('Label Dimensions: ', dataTest.get_label_shape())

    tf.set_random_seed(1)
    X, T, Y, O = dataTest.get_batch_op()
    is_training = tf.placeholder(tf.bool)

    model, loss, mean_ler = bilstm_model(X, Y, T, is_training)

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
            print('Valid: avg_cost = %.4f' % (valid_loss_val),
                  ', avg_ler = %.4f' % (valid_mean_ler_value))
            metrics['test_time_sec'] = metrics['test_time_sec'] + (time.time() - t0)
            metrics['avg_loss'] += valid_loss_val
            metrics['avg_ler'] += valid_mean_ler_value
            str_decoded = convert_2_str(output)
            print('Decoded: ', str_decoded)
            print('Original: ', Original_transcript[0].decode('utf-8'))
        except tf.errors.OutOfRangeError:
            print("Finished Decoding!!!")
            break

    print('Decoded Time = %.4fs, avg_loss = %.4f, avg_ler = %.4f' % (
        metrics['test_time_sec'], metrics['avg_loss'] / global_step, metrics['avg_ler'] / global_step))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read data from featurized mfcc files.")
    parser.add_argument("-i", "--input", required=True,
                        help="List of pickle files containing mfcc")
    parser.add_argument("-m", "--model_dir", required=False, default='.model',
                        help="Directory where trained model files are saved.")
    args = parser.parse_args()

    dataTest = DataSet(args.input, batch_size=1, epochs=1)
    decode(dataTest, args.model_dir)
