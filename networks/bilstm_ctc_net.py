import tensorflow as tf

from .common import (label_error_rate, loss, model, setup_training_network,
                     variable_on_worker_level)


def create_network(features, seq_len, num_classes, is_training):
    '''
    Simple BiLSTM network
    '''
    num_hidden = 500
    num_layers = 1

    # Forward direction cell:
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(
        num_hidden, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

    # Backward direction cell:
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
        num_hidden, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                 cell_bw=lstm_bw_cell,
                                                 inputs=features,
                                                 dtype=tf.float32,
                                                 sequence_length=seq_len)

    shape = tf.shape(features)
    batch_s, max_time_steps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])

    W = variable_on_worker_level(
        'W', [num_hidden, num_classes], tf.contrib.layers.xavier_initializer(uniform=False))

    b = variable_on_worker_level(
        'b', [num_classes], tf.constant_initializer(0.))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))
    return logits
