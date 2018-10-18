import tensorflow as tf
from .tfnetwork import TensorFlowNetwork

class LstmCTCNet(TensorFlowNetwork):
    def __init__(self, dataTrain, datavalid, config):
        TensorFlowNetwork.__init__(self, dataTrain, datavalid, config)

    def create_network(self, features, seq_len, num_classes, is_training):
        '''
        Simple LSTM network
        '''
        num_hidden = 500
        num_layers = 3

        cells = [tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
                for i in range(num_layers)]
        stack = tf.contrib.rnn.MultiRNNCell(cells,
                                            state_is_tuple=True)

        outputs, _ = tf.nn.dynamic_rnn(stack, features, seq_len, dtype=tf.float32)

        shape = tf.shape(features)
        batch_s, _ = shape[0], shape[1]

        outputs = tf.reshape(outputs, [-1, num_hidden])

        W = tf.get_variable(
            'W', [num_hidden, num_classes], initializer = tf.contrib.layers.xavier_initializer(uniform=False))

        b = tf.get_variable('b', [num_classes], initializer = tf.constant_initializer(0.))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))
        return logits
