from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from six.moves import range
from datetime import datetime
# from memn2n.modules import embedding, feedforward, multihead_attention, label_smoothing
from .modules import *

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.
    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        # z = tf.zeros([1, s])
        return tf.concat([z, tf.slice(t, [1, 0], [-1, -1])], 0, name=name)


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


class ProjectionOp:
    """ Single layer perceptron
    Project input tensor on the output dimension
    """
    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        assert len(shape) == 2

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W_t = tf.get_variable(
                'weights',
                shape,
                # initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[0],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )
            self.W = tf.transpose(self.W_t)

    def getWeights(self):
        """ Convenience method for some tf arguments
        """
        return self.W, self.b

    def __call__(self, X):
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b

class SeqN2NDialog(object):
    """End-To-End Memory Network."""

    @staticmethod
    def default_params():
        return {
            "batch_size": 10,
            "vocab_size": 30,
            "sentence_size": 15,
            "embedding_size": 32,
            "num_layers": 2,
            "dropout_rate": 0.1,
            "max_grad_norm": 40.0,
            "nonlin": None,
            "initializer": tf.random_normal_initializer(stddev=0.1),
            "optimizer": tf.train.AdamOptimizer(learning_rate=1e-2),
            "session": tf.Session(),
            "name": 'Attention',
            "task_id": 6
        }

    def __init__(self, batch_size, vocab_size, num_layers, sentence_size, embedding_size, dropout_rate=0.1,
                 max_grad_norm=40.0,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                 session=tf.Session(),
                 name='SeqN2N',
                 candidate_size=15,
                 task_id=6):
        """Creates an End-To-End Memory Network
        Args:
            batch_size: The size of the batch.
            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.
            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).
            candidates_size: The size of candidates
            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).
            embedding_size: The size of the word embedding.
            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.
            nonlin: Non-linearity. Defaults to `None`.
            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.
            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.
            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.
            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.
            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate
        self._sentence_size = sentence_size
        self._candidate_size = candidate_size
        self._embedding_size = embedding_size
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._name = name

        self._build_inputs()

        # define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = "%s_%s_%s_%s/" % ('task',
                                          str(task_id), 'summary_output', timestamp)

        # seq loss
        extended_answers = [tf.ones_like(self._answers[0]) * 2] + self._answers
        go_answers = extended_answers[:-1]

        logits = self._inference(self._stories, go_answers, self._is_training)
        self.logits = logits
        self.preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
        answers_T = tf.transpose(tf.stack(extended_answers[1:], 0))
        self.istarget = tf.to_float(tf.not_equal(answers_T, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, answers_T)) * self.istarget) / (
            tf.reduce_sum(self.istarget))
        self.y_smoothed = label_smoothing(tf.one_hot(answers_T, depth=self._vocab_size))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y_smoothed)
        mean_loss = tf.reduce_sum(loss * self.istarget) / (tf.reduce_sum(self.istarget))
        # loss op
        loss_op = mean_loss

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                          for g, v in grads_and_vars]
        train_op = self._opt.apply_gradients(
            grads_and_vars, name="train_op")

        # predict ops
        self.predict_op = self.preds

        # assign ops
        self.loss_op = loss_op
        self.train_op = train_op

        self.graph_output = self.loss_op

        init_op = tf.initialize_all_variables()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._stories = [tf.placeholder(tf.int32, [None, ]) for _ in range(self._sentence_size)]
        self._answers = [tf.placeholder(tf.int32, [None, ]) for _ in range(self._candidate_size)]
        self._is_training = tf.placeholder(tf.bool, shape=None, name='is_training')

    def _inference(self, stories, answers, is_training):

        with tf.variable_scope("encoder-decoder"):
            def create_rnn_cell():
                encoDecoCell = tf.contrib.rnn.BasicLSTMCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                    self._embedding_size,
                )
                return encoDecoCell

            encoDecoCell = tf.contrib.rnn.MultiRNNCell(
                [create_rnn_cell() for _ in range(self._num_layers)],
            )
            # outputProjection = ProjectionOp(
            #     (self._vocab_size, self._embedding_size),
            #     scope='softmax_projection'
            # )
            logits,_ = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                stories, answers, encoDecoCell, self._vocab_size,
                self._vocab_size, embedding_size=self._embedding_size,
                feed_previous=tf.not_equal(is_training, True)
                # output_projection=outputProjection.getWeights()
            )

        return tf.transpose(tf.stack(logits, 0), (1, 0, 2))

    def batch_fit(self, stories, answers, queries=None):
        # TODO: need to add <S> token to the answer
        """Runs the training algorithm over the passed batch
        Args:
            stories: Tensor (None, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)
        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        stories = np.array(stories, dtype='int32').transpose()[::-1]
        answers = np.array(answers, dtype='int32').transpose()

        feed_dict = {self._stories[i]: stories[i] for i in range(self._sentence_size)}
        feed_dict.update({self._answers[i]: answers[i] for i in range(self._candidate_size)})
        feed_dict.update({self._is_training: True})
        # feed_dict = {self._stories: stories, self._answers: answers, self._is_training: True}

        loss, _ = self._sess.run(
            [self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, stories, queries=None):
        # TODO: need to add <S> token to the answer
        """Predicts answers as one-hot encoding.
        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        answers = np.zeros((len(stories), self._candidate_size), np.int32)
        stories = np.array(stories, dtype='int32').transpose()[::-1]
        answers = np.array(answers, dtype='int32').transpose()
        feed_dict = {self._stories[i]: stories[i] for i in range(self._sentence_size)}
        feed_dict.update({self._answers[i]: answers[i] for i in range(self._candidate_size)})
        feed_dict.update({self._is_training: False})
        return self._sess.run(self.predict_op, feed_dict=feed_dict)


class AttentionModelTest():
    """
    Tests the UnidirectionalRNNEncoder class.
    """

    def __init__(self):
        # super(AttentionModelTest, self).setUp()
        rnn_encoder = SeqN2NDialog
        self.batch_size = 10
        self.sequence_length = 15
        self.mode = tf.contrib.learn.ModeKeys.TRAIN
        self.params = rnn_encoder.default_params()
        self.model = SeqN2NDialog(**self.params)

    def test_encode(self):
        inputs = np.random.random_integers(0, 30, [self.batch_size, self.sequence_length])
        inputs = tf.Variable(initial_value=inputs, dtype=tf.int32)
        encode_fn = self.model._inference
        encoder_output = encode_fn(inputs, inputs, is_training=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            encoder_output_ = sess.run(encoder_output)
        print(encoder_output_.shape)
        # np.testing.assert_array_equal(encoder_output_.shape,
        #                               [self.batch_size, self.sequence_length, 32])

    def test_batch_pred(self):
        inputs = np.random.random_integers(0, 29, [self.batch_size, self.sequence_length])
        inputs2 = np.random.random_integers(0, 29, [self.batch_size, self.sequence_length])
        loss = self.model.batch_fit(inputs, inputs2)
        # loss = self.model.predict(inputs)
        print(inputs2)
        print()
        print(loss)

if __name__ == '__main__':
    model = AttentionModelTest()
    model.test_batch_pred()
    # tf.test.main()