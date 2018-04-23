import tensorflow as tf
import numpy as np



def MakeFancyRNNCell(H, keep_prob, num_layers=1):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers

    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cells = []
    for _ in range(num_layers):
      cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0)
      cell = tf.nn.rnn_cell.DropoutWrapper(
          cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
      cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)


class TextRNN(object):
    """
    A RNN for text classification.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, embedding_size):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("batch_size"):
            self.batch_size = tf.shape(self.input_x)[0]

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W_embed = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W_embed", trainable=False)

            # The result of the embedding operation is a 3-dimensional tensor of shape [None, sequence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.W_embed, self.input_x)

            # Construct RNN/LSTM cell and recurrent layer.
        with tf.name_scope("Recurrent_Layer"):
            self.cell = MakeFancyRNNCell(embedding_size, self.dropout_keep_prob)

            self.unstacked_inputs = tf.unstack(self.embedded_chars, sequence_length, 1)
            # 'final_h' is a tensor of shape [batch_size, cell_state_size]
            # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
            self.outputs, self.final_h = tf.nn.static_rnn(self.cell, self.unstacked_inputs, dtype=tf.float32)
            self.last_outputs = self.outputs[-1]

        with tf.name_scope("Output_Layer"):
            self.W_out = tf.Variable(tf.random_normal([embedding_size, num_classes]), name="W_out")
            self.b_out = tf.Variable(tf.zeros([num_classes,], dtype=tf.float32), name="b_out")
            self.scores = tf.nn.xw_plus_b(self.last_outputs, self.W_out, self.b_out, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
        	# losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses = tf.nn.weighted_cross_entropy_with_logits(logits=self.scores, targets=self.input_y, pos_weight=10)
            self.loss = tf.reduce_mean(losses, name="loss")

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
