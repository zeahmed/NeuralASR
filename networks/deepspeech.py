import tensorflow as tf


def create_model(features, labels, seq_len, num_classes, is_training):

    num_hidden = 500
    num_layers = 1

    # Forward direction cell:
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(
        num_hidden, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    # lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
    #                                             input_keep_prob=1.0 - dropout[3],
    #                                             output_keep_prob=1.0 - dropout[3],
    #                                             seed=FLAGS.random_seed)
    # Backward direction cell:
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
        num_hidden, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    # lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
    #                                             input_keep_prob=1.0 - dropout[4],
    #                                             output_keep_prob=1.0 - dropout[4],
    #                                             seed=FLAGS.random_seed)

    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                 cell_bw=lstm_bw_cell,
                                                 inputs=features,
                                                 dtype=tf.float32,
                                                 sequence_length=seq_len)

    shape = tf.shape(features)
    batch_s, max_time_steps = shape[0], shape[1]

    outputs = tf.reshape(outputs, [-1, num_hidden])

    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))

    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))
    loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits, seq_len))

    model, log_prob = tf.nn.ctc_beam_search_decoder(
        logits, seq_len)  # tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Label Error Rate
    ler = tf.edit_distance(tf.cast(model[0], tf.int32), labels)
    mean_ler = tf.reduce_mean(ler)

    return model[0], loss, mean_ler
