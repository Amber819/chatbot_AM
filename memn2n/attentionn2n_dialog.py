from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from six.moves import range
from datetime import datetime
from memn2n.modules import *

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


class AttentionN2NDialog(object):
    """End-To-End Memory Network."""

    @staticmethod
    def default_params():
        return {
            "batch_size": 10,
            "vocab_size": 40,
            "sentence_size": 10,
            "embedding_size": 32,
            "blocks": 2,
            "num_heads": 2,
            "dropout_rate": 0.1,
            "max_grad_norm": 40.0,
            "nonlin": None,
            "initializer": tf.random_normal_initializer(stddev=0.1),
            "optimizer": tf.train.AdamOptimizer(learning_rate=1e-2),
            "session": tf.Session(),
            "name": 'Attention',
            "task_id": 6
        }

    def __init__(self, batch_size, vocab_size, sentence_size, embedding_size,
                 blocks=6, num_heads=8, dropout_rate=0.1,
                 max_grad_norm=40.0,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                 session=tf.Session(),
                 name='AttentionN2N',
                 candidate_size=29,
                 task_id=6):
        """Creates an End-To-End Full Attention Network

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
        self._blocks = blocks
        self._dropout_rate = dropout_rate
        self._num_heads = num_heads
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
        logits = self._inference(self._stories, self._answers, self._is_training)
        # logits = self._rnn_inference(self._stories, self._answers)
        self.logits = logits
        self.preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
        self.istarget = tf.to_float(tf.not_equal(self._answers, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self._answers)) * self.istarget) / (
            tf.reduce_sum(self.istarget))
        self.y_smoothed = label_smoothing(tf.one_hot(self._answers, depth=self._vocab_size))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y_smoothed)
        mean_loss = tf.reduce_sum(loss * self.istarget) / (tf.reduce_sum(self.istarget))

        # loss op
        loss_op = mean_loss

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v)
                          for g, v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]

        train_op = self._opt.apply_gradients(
            grads_and_vars, name="train_op")

        # predict ops
        self.predict_op = self.preds

        # assign ops
        self.loss_op = loss_op

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.98, epsilon=1e-8)
        # train_op = self.optimizer.minimize(mean_loss)

        self.train_op = train_op

        self.graph_output = self.loss_op

        init_op = tf.initialize_all_variables()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, shape=(None, None), name="stories")
        self._answers = tf.placeholder(tf.int32, shape=(None, self._candidate_size), name="answers")
        self._is_training = tf.placeholder(tf.bool, shape=None, name='is_training')

    def _inference(self, stories, answers, is_training):
        #  Encoder Embedding
        self.enc = embedding(stories,
                             vocab_size=self._vocab_size,
                             num_units=self._embedding_size,
                             scale=True,
                             scope="embed")
        ## Positional Encoding
        self.enc += embedding(
            tf.tile(tf.expand_dims(tf.range(tf.shape(stories)[1]), 0), [tf.shape(stories)[0], 1]),
            vocab_size=self._sentence_size,
            num_units=self._embedding_size,
            zero_pad=False,
            scale=False,
            scope="enc_pe")

        ## Dropout
        self.enc = tf.layers.dropout(self.enc,
                                     rate=self._dropout_rate,
                                     training=is_training)

        #  Decoder Embedding
        self.decoder_inputs = tf.concat((tf.ones_like(answers[:, :1]) * 2, answers[:, :-1]), -1)  # 2:<S>
        self.dec = embedding(self.decoder_inputs,
                             vocab_size=self._vocab_size,
                             num_units=self._embedding_size,
                             scale=True,
                             reuse=True,
                             scope="embed")

        ## Positional Encoding
        self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0),
                                      [tf.shape(self.decoder_inputs)[0], 1]),
                              vocab_size=self._candidate_size,
                              num_units=self._embedding_size,
                              zero_pad=False,
                              scale=False,
                              scope="dec_pe")

        ## Dropout
        self.dec = tf.layers.dropout(self.dec,
                                     rate=self._dropout_rate,
                                     training=is_training)

        with tf.variable_scope("encoder"):
            ## Blocks
            for i in range(self._blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                   keys=self.enc,
                                                   num_units=self._embedding_size,
                                                   num_heads=self._num_heads,
                                                   dropout_rate=self._dropout_rate,
                                                   is_training=is_training,
                                                   causality=False)

                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4 * self._embedding_size, self._embedding_size])


        with tf.variable_scope("decoder"):
            ## Blocks
            for i in range(self._blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                   keys=self.dec,
                                                   num_units=self._embedding_size,
                                                   num_heads=self._num_heads,
                                                   dropout_rate=self._dropout_rate,
                                                   is_training=is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    ## Multihead Attention ( vanilla attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                   keys=self.enc,
                                                   num_units=self._embedding_size,
                                                   num_heads=self._num_heads,
                                                   dropout_rate=self._dropout_rate,
                                                   is_training=is_training,
                                                   causality=False,
                                                   scope="vanilla_attention")

                    ## Feed Forward
                    self.dec = feedforward(self.dec, num_units=[4 * self._embedding_size, self._embedding_size])

        logits = tf.layers.dense(self.dec, self._vocab_size)
        return logits

    def _rnn_inference(self, stories, answers):
        decoder_inputs = tf.concat((tf.ones_like(answers[:, :1]) * 2, answers[:, :-1]), -1)
        embeddings = tf.Variable(tf.random_uniform([self._vocab_size, self._embedding_size], -1.0, 1.0), dtype=tf.float32)

        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, stories)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

        encoder_cell = tf.contrib.rnn.LSTMCell(self._embedding_size)

        self.encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
            encoder_cell, encoder_inputs_embedded,
            dtype=tf.float32, scope="plain_encoder"
        )
        decoder_cell = tf.contrib.rnn.LSTMCell(self._embedding_size)

        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
            decoder_cell, decoder_inputs_embedded,

            initial_state=encoder_final_state,

            dtype=tf.float32, scope="plain_decoder",
        )

        decoder_logits = tf.contrib.layers.linear(decoder_outputs, self._vocab_size)
        return decoder_logits

    def batch_fit(self, stories, answers, queries=None):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        X = np.matrix(stories, dtype='int32')
        Y = np.matrix(answers, dtype='int32')

        feed_dict = {self._stories: X, self._answers: Y, self._is_training: True}
        loss, _ = self._sess.run(
            [self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, stories, queries=None):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        # TODO:split the \s symbol get right sentence indexes
        stories = np.matrix(stories, dtype='int32')
        ### Autoregressive inference
        preds = np.zeros((stories.shape[0], self._candidate_size), np.int32)
        for j in range(self._candidate_size):
            feed_dict = {self._stories: stories, self._answers: preds, self._is_training: False}
            _preds = self._sess.run(self.predict_op, feed_dict=feed_dict)
            preds[:, j] = _preds[:, j]

        return preds


class AttentionModelTest():
    """
    Tests the UnidirectionalRNNEncoder class.
    """

    def __init__(self):
        # super(AttentionModelTest, self).setUp()
        rnn_encoder = AttentionN2NDialog
        self.batch_size = 10
        self.sequence_length = 15
        self.mode = tf.contrib.learn.ModeKeys.TRAIN
        self.params = rnn_encoder.default_params()
        self.model = AttentionN2NDialog(**self.params)
        self.encode_fn = self.model._inference
    def test_encode(self):
        inputs = np.random.random_integers(0, 30, [self.batch_size, self.sequence_length])
        inputs = tf.Variable(initial_value=inputs, dtype=tf.int32)
        encoder_output = self.encode_fn(inputs, inputs, is_training=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            encoder_output_ = sess.run(encoder_output)
        print(encoder_output_.shape)
        # np.testing.assert_array_equal(encoder_output_.shape,
        #                               [self.batch_size, self.sequence_length, 32])

    def test_batch_pred(self):
        inputs = np.random.random_integers(0, 30, [self.batch_size, self.sequence_length])
        inputs2 = np.random.random_integers(0, 30, [self.batch_size, 29])
        loss = self.model.batch_fit(inputs, inputs2)
        # loss = self.model.predict(inputs)
        print(loss.shape)

if __name__ == '__main__':
    model = AttentionModelTest()
    model.test_batch_pred()
    # tf.test.main()