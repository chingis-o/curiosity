class StatePredictor(object):
    '''
    Loss is normalized across spatial dimension (42x42), but not across batches.
    It is unlike ICM where no normalization is there across 288 spatial dimension
    and neither across batches.
    '''

    def __init__(self, ob_space, ac_space, designHead='universe', unsupType='state'):
        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])
        self.stateAenc = unsupType == 'stateAenc'

        # feature encoding: phi1: [None, LEN]
        if designHead == 'universe':
            phi1 = universeHead(phi1)
            if self.stateAenc:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    phi2_aenc = universeHead(phi2)
        elif 'tile' in designHead:  # for mario tiles
            phi1 = universeHead(phi1, nConvs=2)
            if self.stateAenc:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    phi2_aenc = universeHead(phi2)
        else:
            print('Only universe designHead implemented for state prediction baseline.')
            exit(1)

        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        f = tf.concat(1, [phi1, asample])
        f = tf.nn.relu(linear(f, phi1.get_shape()[1].value, "f1", normalized_columns_initializer(0.01)))
        if 'tile' in designHead:
            f = inverseUniverseHead(f, input_shape, nConvs=2)
        else:
            f = inverseUniverseHead(f, input_shape)
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        if self.stateAenc:
            self.aencBonus = 0.5 * tf.reduce_mean(tf.square(tf.subtract(phi1, phi2_aenc)), name='aencBonus')
        self.predstate = phi1

        # variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_state(self, s1, asample):
        '''
        returns state predicted by forward model
            input: s1: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: s2: [h, w, ch]
        '''
        sess = tf.get_default_session()
        return sess.run(self.predstate, {self.s1: [s1],
                                            self.asample: [asample]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        sess = tf.get_default_session()
        bonus = self.aencBonus if self.stateAenc else self.forwardloss
        error = sess.run(bonus,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        # print('ErrorF: ', error)
        error = error * constants['PREDICTION_BETA']
        return error