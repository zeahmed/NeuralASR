import tensorflow as tf

def create_model(dataset, features, labels, seq_len, is_training):

    num_hidden=100
    num_layers=1
    num_classes = ord('z') - ord('a') + 1 + 1 + 1

    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)

    outputs, _ = tf.nn.dynamic_rnn(stack, features, seq_len, dtype=tf.float32)

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
    
    model, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len) #tf.nn.ctc_greedy_decoder(logits, seq_len)
    return model[0], loss