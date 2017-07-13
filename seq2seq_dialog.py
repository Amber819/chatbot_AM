from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_dialog_task, vectorize_data, load_candidates, \
    vectorize_seq2seq_fix, tokenize, vectorize_seq2seq, vectorize_candidates
import metrics
from memn2n import SeqN2NDialog
from itertools import chain
from six.moves import range, reduce
import sys
import tensorflow as tf
import numpy as np
import os

tf.flags.DEFINE_float("learning_rate", 0.001,
                      "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10,
                        "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20,
                        "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("task_id", 6, "bAbI task id, 1 <= id <= 6")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/dialog-bAbI-tasks/",
                       "Directory containing bAbI tasks")
tf.flags.DEFINE_string("model_dir", "model/",
                       "Directory containing memn2n model checkpoints")
tf.flags.DEFINE_boolean('train', True, 'if True, begin to train')
tf.flags.DEFINE_boolean('interactive', False, 'if True, interactive')
FLAGS = tf.flags.FLAGS
print("Started Task:", FLAGS.task_id)


class chatBot(object):
    def __init__(self, data_dir, model_dir, task_id, isInteractive=True, random_state=None,
                 batch_size=32, learning_rate=0.001, epsilon=1e-8, max_grad_norm=40.0, evaluation_interval=1,
                 epochs=200, embedding_size=300, sentence_size=20):
        self.data_dir = data_dir
        self.task_id = task_id
        self.model_dir = model_dir
        # self.isTrain=isTrain
        self.isInteractive = isInteractive
        self.random_state = random_state
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.evaluation_interval = evaluation_interval
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.sentence_size = sentence_size
        candidates, self.candid2indx = load_candidates(
            self.data_dir, self.task_id)
        self.n_cand = len(candidates)
        print("Candidate Size", self.n_cand)
        self.indx2candid = dict(
            (self.candid2indx[key], key) for key in self.candid2indx)
        # task data
        self.trainData, self.testData, self.valData = load_dialog_task(
            self.data_dir, self.task_id, self.candid2indx, False)
        data = self.trainData + self.testData + self.valData
        self.build_vocab(data, candidates)
        self.candidates_vec=vectorize_candidates(candidates,self.word_idx, self.candidate_sentence_size)
        # self.candidates_vec = vectorize_seq2seq_candidates(
        #     candidates, self.word_idx, self.candidate_sentence_size)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.sess = tf.Session()

        self.model = SeqN2NDialog(self.batch_size, self.vocab_size, 1, self.sentence_size, self.embedding_size,
                                        dropout_rate=0.1,
                                        max_grad_norm=40.0,
                                        nonlin=None,
                                        optimizer=optimizer,
                                        session=self.sess,
                                        name='MemN2N',
                                        candidate_size=self.candidate_sentence_size,
                                        task_id=6)
        self.saver = tf.train.Saver(max_to_keep=50)

        self.summary_writer = tf.summary.FileWriter(
            self.model.root_dir, self.model.graph_output.graph)

    def build_vocab(self, data, candidates):
        """0：<PAD>, 1:<UNK>, 2:<S>, 3:</S>"""
        vocab = reduce(lambda x, y: x | y, (set(
            list(chain.from_iterable(s)) + q) for s, q, a in data))
        vocab |= reduce(lambda x, y: x | y, (set(candidate)
                                             for candidate in candidates))
        extra = ['<PAD>', '<UNK>', '<S>', '</S>']
        vocab -= set(extra)
        vocab = sorted(vocab)   # the built-in sorted function is guaranteed to be stable
        vocab = extra + vocab
        self.word_idx = dict((c, i) for i, c in enumerate(vocab))
        self.idx_word = dict((i, c) for i, c in enumerate(vocab))
        self.candidate_sentence_size = max(map(len, candidates)) + 1  # requested for </S> symbol
        self.vocab_size = len(self.word_idx)  # +1 for nil word
        # params
        print("vocab size:", self.vocab_size)
        print("Longest sentence length", self.sentence_size)
        print("Longest candidate sentence length",
              self.candidate_sentence_size)

    def interactive(self):
        context = []
        u = None
        r = None
        nid = 1
        while True:
            line = input('--> ').strip().lower()
            if line == 'exit':
                break
            if line == 'restart':
                context = []
                nid = 1
                print("clear memory")
                continue
            u = tokenize(line)
            data = [(context, u, -1)]
            s, q, a = vectorize_data(
                data, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
            preds = self.model.predict(s, q)
            r = self.indx2candid[preds[0]]
            print(r)
            r = tokenize(r)
            u.append('$u')
            u.append('#' + str(nid))
            r.append('$r')
            r.append('#' + str(nid))
            context.append(u)
            context.append(r)
            nid += 1

    def train(self):
        trainS, trainA = vectorize_seq2seq_fix(
            self.trainData, self.word_idx, self.sentence_size, self.batch_size, self.candidate_sentence_size)
        valS, valA = vectorize_seq2seq_fix(
            self.valData, self.word_idx, self.sentence_size, self.batch_size, self.candidate_sentence_size)

        n_train = len(trainS)
        n_val = len(valS)
        print("Training Size", n_train)
        print("Validation Size", n_val)
        tf.set_random_seed(self.random_state)
        batches = zip(range(0, n_train - self.batch_size, self.batch_size),
                      range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]
        best_validation_accuracy = 0

        for t in range(1, self.epochs + 1):
            np.random.shuffle(batches)
            total_cost = 0.0
            for start, end in batches:
                s = trainS[start:end]
                a = trainA[start:end]
                cost_t = self.model.batch_fit(s, a)
                total_cost += cost_t

            if t % self.evaluation_interval == 0:
                train_preds = self.batch_predict(trainS, n_train)
                val_preds = self.batch_predict(valS, n_val)
                train_acc = metrics.bleu_score(
                    np.array(train_preds), trainA)
                val_acc = metrics.bleu_score(val_preds, valA)
                self.sample_output(valS[:30], valA[:30], val_preds[:30])

                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('Training bleu:', train_acc)
                print('Validation bleu:', val_acc)
                print('-----------------------')

                # write summary
                train_acc_summary = tf.summary.scalar(
                    'task_' + str(self.task_id) + '/' + 'train_acc', tf.constant((train_acc), dtype=tf.float32))
                val_acc_summary = tf.summary.scalar(
                    'task_' + str(self.task_id) + '/' + 'val_acc', tf.constant((val_acc), dtype=tf.float32))
                merged_summary = tf.summary.merge(
                    [train_acc_summary, val_acc_summary])
                summary_str = self.sess.run(merged_summary)
                self.summary_writer.add_summary(summary_str, t)
                self.summary_writer.flush()

                if val_acc > best_validation_accuracy:
                    best_validation_accuracy = val_acc
                    self.saver.save(self.sess, self.model_dir +
                                    'model.ckpt', global_step=t)

    def test(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
        if self.isInteractive:
            self.interactive()
        else:
            testS, testQ, testA = vectorize_seq2seq(
                self.testData, self.word_idx, self.sentence_size, self.batch_size, self.candidate_sentence_size)
            n_test = len(testS)
            print("Testing Size", n_test)
            test_preds = self.batch_predict(testS, n_test)
            test_acc = metrics.bleu_score(test_preds, testA)
            print("Testing bleu:", test_acc)

    def sample_output(self, H, S, A):
        for h, s, a in zip(H, S, A):
            h = [self.idx_word[idx] for idx in h]
            s = [self.idx_word[idx] for idx in s]
            a = [self.idx_word[idx] for idx in a]
            print('--History--:%s\n--answer--:%s\n--Got--:%s\n\n' % (' '.join(h), ' '.join(s), ' '.join(a)))

    def batch_predict(self, S, n):
        preds = []
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            s = S[start:end]
            pred = self.model.predict(s)
            preds += list(pred)
        ret = []
        for pred in preds:
            try:
                index = pred.tolist().index(3)
            except:
                index = len(pred)
            ret.append(pred[:index])
        return ret

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':
    model_dir = "task" + str(FLAGS.task_id) + "_" + FLAGS.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    chatbot = chatBot(FLAGS.data_dir, model_dir, FLAGS.task_id,
                      isInteractive=FLAGS.interactive, batch_size=FLAGS.batch_size)
    # chatbot.run()
    if FLAGS.train:
        chatbot.train()
    else:
        chatbot.test()
    chatbot.close_session()
