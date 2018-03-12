import tensorflow as tf


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
    # mom_opt = tf.train.MomentumOptimizer(
    #     learning_rate=learning_rate, momentum=0.9)
    # gradients, variables = zip(*mom_opt.compute_gradients(loss))
    # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    # optimizer = mom_opt.apply_gradients(zip(gradients, variables))
    return optimizer
