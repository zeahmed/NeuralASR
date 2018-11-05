import os
import numpy as np
import tensorflow as tf

from .tfnetwork import TensorFlowNetwork
from logger import get_logger


class LAS(TensorFlowNetwork):
    def __init__(self, config, fortraining=False):
        TensorFlowNetwork.__init__(self, config, fortraining, False)

    def pad(self, encoder_inputs):
        paddings = tf.constant([[0, 0], [0, 1],[0, 0]])
        return tf.pad(encoder_inputs, paddings, "CONSTANT")

    def create_network(self, features, labels, seq_len, labels_len, num_classes, is_training):
        '''
        This network is similar Listen, Attend and Spell network https://arxiv.org/pdf/1508.01211.pdf.
        '''

        batch_size = tf.shape(features)[0]
        num_hidden = 250
        encoder_inputs = features
        for i in range(4):
            encoder_inputs = tf.cond(tf.shape(encoder_inputs)[1] % 2 > 0,
                        lambda: self.pad(encoder_inputs), lambda: encoder_inputs)
            length = encoder_inputs.get_shape().as_list()[1]
            # Forward direction cell:
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(
                num_hidden, forget_bias=1.0, state_is_tuple=True, name='fw_'+str(i), reuse=tf.get_variable_scope().reuse)

            # Backward direction cell:
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
                num_hidden, forget_bias=1.0, state_is_tuple=True, name='bw_'+str(i),reuse=tf.get_variable_scope().reuse)

            encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                              cell_bw=lstm_bw_cell,
                                                                              inputs=encoder_inputs,
                                                                              dtype=tf.float32,
                                                                              sequence_length=length)
            encoder_outputs = tf.concat(encoder_outputs, axis=2)
            encoder_inputs = encoder_outputs[:,::2,:], encoder_outputs[:,1::2,:]
            encoder_inputs = tf.concat(encoder_inputs, axis=2)

        encoder_states = tf.nn.rnn_cell.LSTMStateTuple(tf.concat([encoder_states[0].c, encoder_states[1].c], axis=1), 
                            tf.concat([encoder_states[0].h, encoder_states[1].h], axis=1))
        beam_width = 1 if self.fortraining else 1000
        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
            encoder_outputs, multiplier=beam_width, name='enc_tile_batch')
        tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
            encoder_states, multiplier=beam_width,  name='state_tile_batch')

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=2 * num_hidden, memory=tiled_encoder_outputs, name='BahdanauAttention')
                #, memory_sequence_length=dims[1])

        cell = tf.contrib.rnn.BasicLSTMCell(
                2 * num_hidden, forget_bias=1.0, state_is_tuple=True, name='decoder_lstm', reuse=tf.get_variable_scope().reuse)

        attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                            attention_layer_size=num_hidden, name='AttentionWrapper')

        projection_layer = tf.layers.Dense(num_classes, name='projection_layer')
        init_state = attn_cell.zero_state(dtype=tf.float32, batch_size=batch_size*beam_width).clone(cell_state = tiled_encoder_final_state)
        embeddings = tf.one_hot(list(range(self.config.symbols.counter)), self.config.symbols.counter)
        if self.fortraining:
            max_label_len = tf.fill([batch_size],tf.shape(labels)[1])
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                            tf.nn.embedding_lookup(embeddings, labels), max_label_len, embeddings, 0.3)
            # helper = tf.contrib.seq2seq.TrainingHelper(
            #                 tf.nn.embedding_lookup(embeddings, labels), max_label_len)
            max_iterations = tf.shape(labels)[1]
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell, helper=helper,
                initial_state=init_state, output_layer = projection_layer)
        else:
            start_id = self.config.symbols.get_id(self.config.start_marker)
            end_id = self.config.symbols.get_id(self.config.end_marker)
            #helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, tf.fill([batch_size], start_id), end_id)
            # helper =  tf.contrib.seq2seq.InferenceHelper(
            #         sample_fn=lambda outputs: tf.argmax(outputs,axis=1),
            #         sample_shape=[],  # again because dim=1
            #         sample_dtype=tf.int64,
            #         start_inputs=tf.nn.embedding_lookup(embeddings, tf.fill([batch_size], start_id)),
            #         end_fn=lambda sample_ids: tf.equal(sample_ids, tf.to_int64(tf.fill(tf.shape(sample_ids), end_id))),
            #         next_inputs_fn=lambda sample_ids: tf.nn.embedding_lookup(embeddings, sample_ids))
            # decoder = tf.contrib.seq2seq.BasicDecoder(
            #     cell=attn_cell, helper=helper,
            #     initial_state=init_state, output_layer = projection_layer)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell = attn_cell,
                embedding = embeddings,
                start_tokens = tf.tile([start_id], [batch_size]),
                end_token = end_id,
                initial_state = init_state,
                beam_width = beam_width,
                output_layer = projection_layer,
                length_penalty_weight = 0.5)
            max_iterations = 100

        outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=False, maximum_iterations=max_iterations)

        weights=tf.to_float(tf.sequence_mask(labels_len, tf.shape(labels)[1]))
        if self.fortraining:
            logits = outputs[0].rnn_output
            model = tf.argmax(outputs[0].rnn_output, axis=2)# outputs[0].sample_id
            model = tf.multiply(model, tf.to_int64(weights))
            #labels = tf.Print(labels, [tf.shape(logits), tf.shape(labels), labels[0], model[0]], "Test: ", summarize=100)
        else:
            logits = outputs[0].beam_search_decoder_output.scores
            model = outputs[0].predicted_ids[:,:,0]
        loss = tf.contrib.seq2seq.sequence_loss(logits, labels, weights=weights)
        ler = self.create_metric(tf.contrib.layers.dense_to_sparse(model),
                    tf.contrib.layers.dense_to_sparse(labels))
        return logits, loss, model, None, ler

    def validate(self, mfccs, labels, seq_len, labels_len):
        feed_dict={self.is_training: False, self.features: mfccs, self.labels: labels, self.seq_len: seq_len, self.labels_len: labels_len}
        return self.sess.run([self.loss, self.mean_ler], feed_dict=feed_dict)

    def evaluate(self, mfccs, labels, seq_len, labels_len):
        feed_dict = {self.is_training: False,
                     self.features: mfccs, self.labels: labels, self.seq_len: seq_len, self.labels_len: labels_len}
        return self.sess.run([self.model, self.loss, self.mean_ler], feed_dict=feed_dict)

    def decode(self, mfccs, seq_len):
        feed_dict = {self.is_training: False, self.features: mfccs, self.seq_len: seq_len}
        return self.sess.run(self.model, feed_dict=feed_dict)[0]

    def train(self, mfccs, labels, seq_len, labels_len):
        self.global_step += 1
        feed_dict={self.is_training: True, self.features: mfccs, self.labels: labels, self.seq_len: seq_len, self.labels_len: labels_len}
        _, loss_val, mean_ler_value = self.sess.run(
            [self.optimizer, self.loss, self.mean_ler], feed_dict=feed_dict)
        return loss_val, mean_ler_value
