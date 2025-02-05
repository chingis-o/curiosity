from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from constants import constants


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space, designHead='universe'):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x')
        size = 256
        if designHead == 'nips':
            x = nipsHead(x)
        elif designHead == 'nature':
            x = natureHead(x)
        elif designHead == 'doom':
            x = doomHead(x)
        elif 'tile' in designHead:
            x = universeHead(x, nConvs=2)
        else:
            x = universeHead(x)

        # introduce a "fake" batch dimension of 1 to do LSTM over time dim
        x = tf.expand_dims(x, [0])
        lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c_in')
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h_in')
        self.state_in = [c_in, h_in]

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # [0, :] means pick action of first state from batch. Hardcoded b/c
        # batch=1 during rollout collection. Its not used during batch training.
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        # tf.add_to_collection('probs', self.probs)
        # tf.add_to_collection('sample', self.sample)
        # tf.add_to_collection('state_out_0', self.state_out[0])
        # tf.add_to_collection('state_out_1', self.state_out[1])
        # tf.add_to_collection('vf', self.vf)

    def get_initial_features(self):
        # Call this function to get reseted lstm memory cells
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def act_inference(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]

