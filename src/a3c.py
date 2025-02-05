from __future__ import print_function
import tensorflow as tf
from model import LSTMPolicy, StateActionPredictor, StatePredictor
from constants import constants

import runner_thread as RunnerThread

class A3C(object):
    def __init__(self, env, task, visualise, unsupType, envWrap=False, designHead='universe', noReward=False):
        """
        An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
        Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
        But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
        should be computed.
        """
        self.task = task
        self.unsup = unsupType is not None
        self.envWrap = envWrap
        self.env = env

        predictor = None
        numaction = env.action_space.n
        worker_device = "/job:worker/task:{}/cpu:0".format(task)

        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, numaction, designHead)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
                if self.unsup:
                    with tf.variable_scope("predictor"):
                        if 'state' in unsupType:
                            self.ap_network = StatePredictor(env.observation_space.shape, numaction, designHead, unsupType)
                        else:
                            self.ap_network = StateActionPredictor(env.observation_space.shape, numaction, designHead)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, numaction, designHead)
                pi.global_step = self.global_step
                if self.unsup:
                    with tf.variable_scope("predictor"):
                        if 'state' in unsupType:
                            self.local_ap_network = predictor = StatePredictor(env.observation_space.shape, numaction, designHead, unsupType)
                        else:
                            self.local_ap_network = predictor = StateActionPredictor(env.observation_space.shape, numaction, designHead)

            # Computing a3c loss: https://arxiv.org/abs/1506.02438
            self.ac = tf.placeholder(tf.float32, [None, numaction], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")
            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)
            # 1) the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_mean(tf.reduce_sum(log_prob_tf * self.ac, 1) * self.adv)  # Eq (19)
            # 2) loss of value function: l2_loss = (x-y)^2/2
            vf_loss = 0.5 * tf.reduce_mean(tf.square(pi.vf - self.r))  # Eq (28)
            # 3) entropy to ensure randomness
            entropy = - tf.reduce_mean(tf.reduce_sum(prob_tf * log_prob_tf, 1))
            # final a3c loss: lr of critic is half of actor
            self.loss = pi_loss + 0.5 * vf_loss - entropy * constants['ENTROPY_BETA']

            # compute gradients
            grads = tf.gradients(self.loss * 20.0, pi.var_list)  # batchsize=20. Factored out to make hyperparams not depend on it.

            # computing predictor loss
            if self.unsup:
                if 'state' in unsupType:
                    self.predloss = constants['PREDICTION_LR_SCALE'] * predictor.forwardloss
                else:
                    self.predloss = constants['PREDICTION_LR_SCALE'] * (predictor.invloss * (1-constants['FORWARD_LOSS_WT']) +
                                                                    predictor.forwardloss * constants['FORWARD_LOSS_WT'])
                predgrads = tf.gradients(self.predloss * 20.0, predictor.var_list)  # batchsize=20. Factored out to make hyperparams not depend on it.

                # do not backprop to policy
                if constants['POLICY_NO_BACKPROP_STEPS'] > 0:
                    grads = [tf.scalar_mul(tf.to_float(tf.greater(self.global_step, constants['POLICY_NO_BACKPROP_STEPS'])), grads_i)
                                    for grads_i in grads]


            self.runner = RunnerThread(env, pi, constants['ROLLOUT_MAXLEN'], visualise,
                                        predictor, envWrap, noReward)

            # storing summaries
            bs = tf.to_float(tf.shape(pi.x)[0])
            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss)
                tf.summary.scalar("model/value_loss", vf_loss)
                tf.summary.scalar("model/entropy", entropy)
                tf.summary.image("model/state", pi.x)  # max_outputs=10
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                if self.unsup:
                    tf.summary.scalar("model/predloss", self.predloss)
                    if 'action' in unsupType:
                        tf.summary.scalar("model/inv_loss", predictor.invloss)
                        tf.summary.scalar("model/forward_loss", predictor.forwardloss)
                    tf.summary.scalar("model/predgrad_global_norm", tf.global_norm(predgrads))
                    tf.summary.scalar("model/predvar_global_norm", tf.global_norm(predictor.var_list))
                self.summary_op = tf.summary.merge_all()
            else:
                tf.scalar_summary("model/policy_loss", pi_loss)
                tf.scalar_summary("model/value_loss", vf_loss)
                tf.scalar_summary("model/entropy", entropy)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                if self.unsup:
                    tf.scalar_summary("model/predloss", self.predloss)
                    if 'action' in unsupType:
                        tf.scalar_summary("model/inv_loss", predictor.invloss)
                        tf.scalar_summary("model/forward_loss", predictor.forwardloss)
                    tf.scalar_summary("model/predgrad_global_norm", tf.global_norm(predgrads))
                    tf.scalar_summary("model/predvar_global_norm", tf.global_norm(predictor.var_list))
                self.summary_op = tf.merge_all_summaries()

            # clip gradients
            grads, _ = tf.clip_by_global_norm(grads, constants['GRAD_NORM_CLIP'])
            grads_and_vars = list(zip(grads, self.network.var_list))
            if self.unsup:
                predgrads, _ = tf.clip_by_global_norm(predgrads, constants['GRAD_NORM_CLIP'])
                pred_grads_and_vars = list(zip(predgrads, self.ap_network.var_list))
                grads_and_vars = grads_and_vars + pred_grads_and_vars

            # update global step by batch size
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            # TODO: make optimizer global shared, if needed
            print("Optimizer: ADAM with lr: %f" % (constants['LEARNING_RATE']))
            print("Input observation shape: ",env.observation_space.shape)
            opt = tf.train.AdamOptimizer(constants['LEARNING_RATE'])
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)

            # copy weights from the parameter server to the local model
            sync_var_list = [v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)]
            if self.unsup:
                sync_var_list += [v1.assign(v2) for v1, v2 in zip(predictor.var_list, self.ap_network.var_list)]
            self.sync = tf.group(*sync_var_list)

            # initialize extras
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer