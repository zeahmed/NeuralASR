import tensorflow as tf

from .tfnetwork import TensorFlowNetwork


class BiLstmCTCNet(TensorFlowNetwork):
    def __init__(self, config, fortraining=False):
        TensorFlowNetwork.__init__(self, config, fortraining)

    def create_network(self, features, labels, seq_len, num_classes, is_training):
        '''
        Simple BiLSTM network
        '''
        num_hidden = 500

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
        batch_s = shape[0]

        outputs = tf.reshape(outputs, [-1, num_hidden])

        W = tf.get_variable(
            'W', [num_hidden, num_classes], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        b = tf.get_variable(
            'b', [num_classes], initializer=tf.constant_initializer(0.))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))
        loss = self.create_loss(logits, labels, seq_len)
        model, prob = self.create_model(logits, seq_len)
        ler = self.create_metric(model, labels)
        return logits, loss, model, prob, ler
