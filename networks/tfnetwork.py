import os
import shutil
import time
from abc import ABC, abstractmethod

import tensorflow as tf

from .network import Network


class TensorFlowNetwork(Network):
    def __init__(self, config, fortraining=True):
        Network.__init__(self)
        self.config = config
        num_classes=config.symbols.counter
        num_gpus=config.num_gpus
        learningrate=config.learningrate
        tf.set_random_seed(1)
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.X = tf.placeholder(tf.float32,shape = [None, None, config.feature_size], name="X")
        self.T = tf.placeholder(tf.int32,shape=[None], name="T")
        self.Y = tf.sparse_placeholder(tf.int32, name="Y")

        if fortraining:
            self.optimizer, self.loss, self.mean_ler = self.setup_training_network(self.X, self.Y, self.T, num_classes, num_gpus, learningrate, self.is_training)
            self.write_config()
            self.config.symbols.write(os.path.join(
                self.config.model_dir, os.path.basename(self.config.sym_file)))
        else:
            self.logits = self.create_network(self.X, self.T, config.symbols.counter, self.is_training)
            self.loss = self.create_loss(self.logits, self.Y, self.T)
            self.model, self.log_prob = self.create_model(self.logits, self.T)
            self.mean_ler = self.create_metric(self.model, self.Y)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(init)

        self.global_step = self.config.start_step
        if fortraining:
            step = self.global_step
        else:
            step = 1
        self.load_model(step, self.sess, self.saver, self.config.model_dir)

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

    def save_step(self):
        self.saver.save(self.sess, os.path.join(self.config.model_dir, 'model'),
                global_step=self.global_step)

    def validation_step(self, mfccs, labels, seq_len):
        feed_dict={self.is_training: False, self.X: mfccs, self.Y: labels, self.T: seq_len}
        return self.sess.run([self.loss, self.mean_ler], feed_dict=feed_dict)

    def evaluate(self, mfccs, labels, seq_len):
        feed_dict={self.is_training: False, self.X: mfccs, self.Y: labels, self.T: seq_len}
        return self.sess.run([self.model, self.loss, self.mean_ler], feed_dict=feed_dict)

    def decode_step(self, mfccs, seq_len):
        feed_dict={self.is_training: False, self.X: mfccs, self.T: seq_len}
        return self.sess.run(self.model, feed_dict=feed_dict)

    def train_step(self, mfccs, labels, seq_len):
        self.global_step += 1
        feed_dict={self.is_training: True, self.X: mfccs, self.Y: labels, self.T: seq_len}
        _, loss_val, mean_ler_value = self.sess.run(
            [self.optimizer, self.loss, self.mean_ler], feed_dict=feed_dict)
        return loss_val, mean_ler_value
