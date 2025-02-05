class StateActionPredictor(object):
    def __init__(self, ob_space, ac_space, designHead='universe'):
        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])

        # feature encoding: phi1, phi2: [None, LEN]
        size = 256
        if designHead == 'nips':
            phi1 = nipsHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = nipsHead(phi2)
        elif designHead == 'nature':
            phi1 = natureHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = natureHead(phi2)
        elif designHead == 'doom':
            phi1 = doomHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = doomHead(phi2)
        elif 'tile' in designHead:
            phi1 = universeHead(phi1, nConvs=2)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = universeHead(phi2, nConvs=2)
        else:
            phi1 = universeHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = universeHead(phi2)

        # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
        g = tf.concat(1,[phi1, phi2])
        g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
        aindex = tf.argmax(asample, axis=1)  # aindex: [batch_size,]
        logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits, aindex), name="invloss")
        self.ainvprobs = tf.nn.softmax(logits, dim=-1)

        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        f = tf.concat(1, [phi1, asample])
        f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
        f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        # self.forwardloss = 0.5 * tf.reduce_mean(tf.sqrt(tf.abs(tf.subtract(f, phi2))), name='forwardloss')
        # self.forwardloss = cosineLoss(f, phi2, name='forwardloss')
        self.forwardloss = self.forwardloss * 288.0  # lenFeatures=288. Factored out to make hyperparams not depend on it.

        # variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_act(self, s1, s2):
        '''
        returns action probability distribution predicted by inverse model
            input: s1,s2: [h, w, ch]
            output: ainvprobs: [ac_space]
        '''
        sess = tf.get_default_session()
        return sess.run(self.ainvprobs, {self.s1: [s1], self.s2: [s2]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        sess = tf.get_default_session()
        # error = sess.run([self.forwardloss, self.invloss],
        #     {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        # print('ErrorF: ', error[0], ' ErrorI:', error[1])
        error = sess.run(self.forwardloss,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        error = error * constants['PREDICTION_BETA']
        return error

