import tensorflow as tf


def create_model(features, seq_len, num_classes, is_training):

    num_hidden = 100
    num_layers = 3

    cells = [tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
             for i in range(num_layers)]
    stack = tf.contrib.rnn.MultiRNNCell(cells,
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

    return logits


def loss(logits, labels, seq_len):
    return tf.reduce_mean(tf.nn.ctc_loss(labels, logits, seq_len))


def model(logits, seq_len):
    model, log_prob = tf.nn.ctc_beam_search_decoder(
        logits, seq_len)  # tf.nn.ctc_greedy_decoder(logits, seq_len)
    return model[0], log_prob


def label_error_rate(model, labels):
    # Label Error Rate
    ler = tf.edit_distance(tf.cast(model, tf.int32), labels)
    mean_ler = tf.reduce_mean(ler)
    return mean_ler


def optimizer(loss, learning_rate):
    adam_opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate)  # .minimize(loss)
    gradients, variables = zip(*adam_opt.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = adam_opt.apply_gradients(zip(gradients, variables))
    return optimizer
