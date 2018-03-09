import tensorflow as tf


def variable_on_worker_level(name, shape, initializer):
    return tf.get_variable(name=name, shape=shape, initializer=initializer)


def create_model(features, labels, seq_len, num_classes, is_training):

    n_input = features.get_shape().as_list()[2]
    dropout = [0.05, 0.05, 0.05, 0., 0., 0.05]
    n_hidden = 1024
    n_hidden_1 = n_hidden
    n_hidden_2 = n_hidden
    n_hidden_5 = n_hidden
    n_cell_dim = n_hidden
    n_hidden_3 = 2 * n_cell_dim
    n_hidden_6 = num_classes
    relu_clip = 20.0
    stddev = 0.046875
    random_seed = 4567

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(features)

    # Reshaping `features` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    features = tf.transpose(features, [1, 0, 2])
    # Reshape to prepare input for first layer
    # (n_steps*batch_size, n_input + 2*n_input*n_context)
    features = tf.reshape(features, [-1, n_input])

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    b1 = variable_on_worker_level(
        'b1', [n_hidden_1], tf.random_normal_initializer(stddev=stddev))
    h1 = variable_on_worker_level(
        'h1', [n_input, n_hidden_1], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(features, h1), b1)), relu_clip)
    layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

    # 2nd layer
    b2 = variable_on_worker_level(
        'b2', [n_hidden_2], tf.random_normal_initializer(stddev=stddev))
    h2 = variable_on_worker_level(
        'h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=stddev))
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
    layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

    # 3rd layer
    b3 = variable_on_worker_level(
        'b3', [n_hidden_3], tf.random_normal_initializer(stddev=stddev))
    h3 = variable_on_worker_level(
        'h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=stddev))
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
    layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

    # Forward direction cell:
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(
        n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                 input_keep_prob=1.0 - dropout[3],
                                                 output_keep_prob=1.0 - dropout[3],
                                                 seed=random_seed)
    # Backward direction cell:
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
        n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                 input_keep_prob=1.0 - dropout[4],
                                                 output_keep_prob=1.0 - dropout[4],
                                                 seed=random_seed)

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                 cell_bw=lstm_bw_cell,
                                                 inputs=layer_3,
                                                 dtype=tf.float32,
                                                 time_major=True,
                                                 sequence_length=seq_len)

    # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
    # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
    outputs = tf.concat(outputs, 2)
    outputs = tf.reshape(outputs, [-1, 2 * n_cell_dim])

    # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    b5 = variable_on_worker_level(
        'b5', [n_hidden_5], tf.random_normal_initializer(stddev=stddev))
    h5 = variable_on_worker_level(
        'h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=stddev))
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_worker_level(
        'b6', [n_hidden_6], tf.random_normal_initializer(stddev=stddev))
    h6 = variable_on_worker_level(
        'h6', [n_hidden_5, n_hidden_6], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    logits = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6], name="logits")

    # Time major
    loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits, seq_len))

    model, log_prob = tf.nn.ctc_beam_search_decoder(
        logits, seq_len)  # tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Label Error Rate
    ler = tf.edit_distance(tf.cast(model[0], tf.int32), labels)
    mean_ler = tf.reduce_mean(ler)

    return model[0], loss, mean_ler, log_prob


def create_optimizer(loss, learning_rate):
    # adam_opt = tf.train.AdamOptimizer(
    #    learning_rate=learning_rate)  # .minimize(loss)
    #gradients, variables = zip(*adam_opt.compute_gradients(loss))
    #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    #optimizer = adam_opt.apply_gradients(zip(gradients, variables))
    mom_opt = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=0.9)
    gradients, variables = zip(*mom_opt.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = mom_opt.apply_gradients(zip(gradients, variables))
    return optimizer
