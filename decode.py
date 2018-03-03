
import numpy as np
import tensorflow as tf

from preprocess.utils import START_INDEX
from common import load_model

def decode_batch(sess, model, feed_dict):
    d = sess.run(model, feed_dict=feed_dict)
    str_decoded = ''.join([chr(x + START_INDEX) for x in np.asarray(d[1])])
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
    return str_decoded

def decode(dataTest, model_dir):
    print('Batch Dimensions: ', dataTest.get_feature_shape())
    print('Label Dimensions: ', dataTest.get_label_shape())

    tf.set_random_seed(1)
    X = tf.placeholder(tf.float32, dataTest.get_feature_shape())
    Y = tf.sparse_placeholder(tf.int32)
    T = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool)

    model, loss = create_model(dataTest, X, Y, T, is_training)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=0.005, momentum=0.9).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    load_model(50, saver, model_dir)
    
    test_time_sec = 0

    dataTest.reset()
    while dataTest.has_more_batches():
        X_batch, Y_batch, seq_len, _ = dataTest.get_next_batch()
        t0 = time.time()
        feed_dict = {X: X_batch, T: seq_len, is_training: False}
        str_decoded = decode_batch(sess, model, feed_dict)
        test_time_sec = test_time_sec + (time.time() - t0)
        print('Decoded: ', str_decoded)
        print('Original: ', original)

    print("Finished Decoding!!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read data from featurized mfcc files.")
    parser.add_argument("-i", "--input", required=True,
                        help="List of pickle files containing mfcc")
    parser.add_argument("-m", "--model_dir", required=False, default='.model',
                        help="Directory to save model files.")    
    args = parser.parse_args()

    dataTest = DataSet(args.input)
    decode(dataTest, args.model_dir)