import os
import shutil
import time
from abc import ABC, abstractmethod

import tensorflow as tf

from .network import Network


class TensorFlowNetwork(Network):
    def __init__(self, dataTrain, datavalid, config):
        Network.__init__(self)
        self.dataTrain = dataTrain
        self.datavalid = datavalid
        self.config = config
        num_classes=config.symbols.counter
        num_gpus=config.num_gpus
        learningrate=config.learningrate
        tf.set_random_seed(1)
        self.is_training = tf.placeholder(tf.bool)

        if datavalid:
            X, T, Y, _ = tf.cond(self.is_training, lambda: dataTrain.get_batch_op(),
                                lambda: datavalid.get_batch_op())
        else:
            X, T, Y, _ = dataTrain.get_batch_op()
        self.optimizer, self.loss, self.mean_ler = self.setup_training_network(X, Y, T, num_classes, num_gpus, learningrate, self.is_training)

    def create_loss(self, logits, labels, seq_len):
        return tf.reduce_mean(tf.nn.ctc_loss(labels, logits, seq_len))

    def create_model(self, logits, seq_len):
        model, log_prob = tf.nn.ctc_beam_search_decoder(
            logits, seq_len)  # tf.nn.ctc_greedy_decoder(logits, seq_len)
        return model[0], log_prob

    def create_metric(self, model, labels):
        # Label Error Rate
        ler = tf.edit_distance(tf.cast(model, tf.int32), labels)
        mean_ler = tf.reduce_mean(ler)
        return mean_ler

    @abstractmethod
    def create_network(self, features, seq_len, num_classes, is_training):
        pass

    def average_gradients(self, tower_grads):
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

    def make_parallel(self, fn, num_gpus, **kwargs):
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


    def setup_training_network(self, X, Y, T, num_classes, num_gpus, learningrate, is_training):
        adam_opt = tf.train.AdamOptimizer(
            learning_rate=learningrate)  # .minimize(loss)
        tower_grads = []

        def create_ops(X, Y, T):
            logits = self.create_network(X, T, num_classes, is_training)
            l = self.create_loss(logits, Y, T)
            m, _ = self.create_model(logits, T)
            ler = self.create_metric(m, Y)
            grads = adam_opt.compute_gradients(l, colocate_gradients_with_ops=True)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)
            return l, ler

        if num_gpus <= 1:
            l, mean_ler = create_ops(X=X, Y=Y, T=T)
        else:
            l, ler = self.make_parallel(
                create_ops, num_gpus=num_gpus, X=X, Y=Y, T=T)
            l = tf.reduce_mean(l)
            mean_ler = tf.reduce_mean(ler)

        grads = self.average_gradients(tower_grads)
        optimizer = adam_opt.apply_gradients(grads)
        return optimizer, l, mean_ler

    def load_model(self, start_epoch, sess, saver, model_dir):
        if start_epoch > 0:
            self.logger.info("Restoring checkpoint: " + model_dir)
            model_file = tf.train.latest_checkpoint(model_dir)
            saver.restore(sess, model_file)
            self.logger.info("Done Restoring checkpoint: " + model_file)
        else:
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            os.makedirs(model_dir)

    def write_config(self):
        model_config = os.path.join(
            self.config.model_dir, os.path.basename(self.config.configfile))
        if os.path.exists(model_config):
            self.logger.warn(
                'Not overwriting. Config file already exists: ' + model_config)
        else:
            self.config.write(model_config)

    def train(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)

            global_step = self.config.start_step
            self.load_model(global_step, sess, saver, self.config.model_dir)
            self.write_config()
            self.config.symbols.write(os.path.join(
                self.config.model_dir, os.path.basename(self.config.sym_file)))

            metrics = {'train_time_sec': 0, 'avg_loss': 0, 'avg_ler': 0}
            # dataTrain.get_num_of_sample() // dataTrain.batch_size
            report_step = self.config.report_step
            while True:
                global_step += 1
                try:
                    t0 = time.time()
                    _, loss_val, mean_ler_value = sess.run(
                        [self.optimizer, self.loss, self.mean_ler], feed_dict={self.is_training: True})
                    metrics['train_time_sec'] += (time.time() - t0)
                    metrics['avg_loss'] += loss_val
                    metrics['avg_ler'] += mean_ler_value
                except tf.errors.OutOfRangeError:
                    break

                if global_step % report_step == 0:
                    saver.save(sess, os.path.join(self.config.model_dir, 'model'),
                            global_step=global_step)
                    self.logger.info('Step: %04d' % (global_step) + ', cost = %.4f' %
                                (metrics['avg_loss'] / report_step) + ', ler = %.4f' % (metrics['avg_ler'] / report_step) +
                                ', time = %.4f' % (metrics['train_time_sec']))
                    metrics['avg_loss'] = 0
                    metrics['avg_ler'] = 0
                    if self.datavalid:
                        valid_loss_val,  valid_mean_ler_value = sess.run(
                            [self.loss, self.mean_ler], feed_dict={self.is_training: False})
                        self.logger.info('Valid: cost = %.4f' % (valid_loss_val) +
                                    ', ler = %.4f' % (valid_mean_ler_value))
            self.logger.info("Finished training!!!")
