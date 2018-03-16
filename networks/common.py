import tensorflow as tf


def variable_on_worker_level(name, shape, initializer):
    return tf.get_variable(name=name, shape=shape, initializer=initializer)


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


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def make_parallel(fn, num_gpus, **kwargs):
    """Parallelize given model on multiple gpu devices.
    adapted from: https://github.com/vahidk/EffectiveTensorflow#make_parallel
    """

    in_splits = {}
    for k, v in kwargs.items():
        if k in ('num_classes', 'is_training'):
            in_splits[k] = [v] * num_gpus
        elif k in ('Y'):
            in_splits[k] = tf.sparse_split(sp_input=v, num_split=num_gpus, axis=0)
        else:
            in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                outputs = fn(**{k: v[i] for k, v in in_splits.items()})
                for o in range(len(outputs)):
                    if o >= len(out_split):
                        out_split.append([])
                    out_split[o].append(outputs[o])

    return [tf.stack(o, axis=0) for o in out_split]


def setup_training_network(create_network, X, Y, T, num_classes, num_gpus, learningrate, is_training):
    adam_opt = tf.train.AdamOptimizer(
        learning_rate=learningrate)  # .minimize(loss)
    tower_grads = []

    def create_ops(X, Y, T):
        logits = create_network(X, T, num_classes, is_training)
        l = loss(logits, Y, T)
        m, log_prob = model(logits, T)
        ler = label_error_rate(m, Y)
        grads = adam_opt.compute_gradients(l, colocate_gradients_with_ops=True)

        # Keep track of the gradients across all towers.
        tower_grads.append(grads)
        return l, ler

    if num_gpus <= 1:
        l, mean_ler = create_ops(X=X, Y=Y, T=T)
    else:
        l, ler = make_parallel(
            create_ops, num_gpus=num_gpus, X=X, Y=Y, T=T)
        l = tf.reduce_mean(l)
        mean_ler = tf.reduce_mean(ler)

    grads = average_gradients(tower_grads)
    optimizer = adam_opt.apply_gradients(grads)
    return optimizer, l, mean_ler
