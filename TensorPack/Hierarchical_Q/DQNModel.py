#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQNModel.py


import abc
import tensorflow as tf
import tensorpack
from tensorpack import ModelDesc
from tensorpack.utils import logger
from tensorpack.tfutils import (
    varreplace, summary, get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
assert tensorpack.tfutils.common.get_tf_version_number() >= 1.2


class Model(ModelDesc):
    learning_rate = 1e-4

    def __init__(self, state_shape, method, num_actions, gamma):
        self.state_shape = state_shape
        self.method = method
        self.num_actions = num_actions
        self.gamma = gamma

    def inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        return [tf.placeholder(tf.float32,
                               (None, 2, self.state_shape[0], self.state_shape[1], self.state_shape[2]),
                               'joint_state'),
                tf.placeholder(tf.int64, (None,), 'action'),
                tf.placeholder(tf.float32, (None,), 'reward'),
                tf.placeholder(tf.bool, (None,), 'isOver'),
                tf.placeholder(tf.bool, (None,), 'comb_mask')]

    # input B * C * D
    # output B * C * 1
    @abc.abstractmethod
    def _get_DQN_prediction_comb(self, state):
        pass

    # input B * N * D
    # output B * N * 1
    @abc.abstractmethod
    def _get_DQN_prediction_fine(self, state):
        pass

    @abc.abstractmethod
    def _get_global_feature(self, joint_state):
        pass

    # decorate the function
    # output : B * A
    @auto_reuse_variable_scope
    def get_DQN_prediction(self, joint_state, comb_mask):
        batch_size = tf.shape(joint_state)[0]
        with tf.variable_scope('dqn_global'):
            global_feature = self.get_global_feature(joint_state)

        comb_mask_idx = tf.cast(tf.where(comb_mask), tf.int32)
        with tf.variable_scope('dqn_comb'):
            q_comb = self._get_DQN_prediction_comb(tf.gather(global_feature, comb_mask_idx[:, 0]))
        q_comb = tf.squeeze(q_comb, -1)
        q_comb = tf.scatter_nd(comb_mask_idx, q_comb, tf.stack([batch_size, q_comb.shape[1]]))

        fine_mask_idx = tf.cast(tf.where(tf.logical_not(comb_mask)), tf.int32)
        state_fine = tf.concat([tf.tile(tf.expand_dims(global_feature, 2), [1, 1, joint_state.shape.as_list()[2], 1]), joint_state], -1)
        state_fine = tf.gather(state_fine[:, 0, :, :], fine_mask_idx[:, 0])
        with tf.variable_scope('dqn_fine'):
            q_fine = self._get_DQN_prediction_fine(state_fine)
        q_fine = tf.squeeze(q_fine, -1)
        q_fine = tf.scatter_nd(fine_mask_idx, q_fine, tf.stack([batch_size, q_fine.shape[1]]))

        larger_dim = max(joint_state.shape.as_list()[1], joint_state.shape.as_list()[2])
        return tf.identity(tf.pad(q_comb, [[0, 0], [0, larger_dim - q_comb.shape.as_list()[1]]]) + tf.pad(q_fine, [[0, 0], [0, larger_dim - q_fine.shape.as_list()[1]]]), name='Qvalue')

    # input :B * COMB * N * D
    # output : B * COMB * D'
    @auto_reuse_variable_scope
    def get_global_feature(self, joint_state):
        return self._get_global_feature(joint_state)

    # joint state: B * 2 * COMB * N * D for now, D = 256
    # dynamic action range
    def build_graph(self, joint_state, action, reward, isOver, comb_mask):
        state = tf.identity(joint_state[:, 0, :, :, :], name='state')
        self.predict_value = self.get_DQN_prediction(state, comb_mask)
        if not get_current_tower_context().is_training:
            return

        reward = tf.clip_by_value(reward, -1, 1)
        next_state = tf.identity(joint_state[:, 1, :, :, :], name='next_state')
        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'), varreplace.freeze_variables(skip_collection=True):
            # we are alternating between comb and fine states
            targetQ_predict_value = self.get_DQN_prediction(next_state, tf.logical_not(comb_mask))    # NxA

        if self.method != 'Double':
            # DQN
            best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        else:
            # Double-DQN
            next_predict_value = self.get_DQN_prediction(next_state, tf.logical_not(comb_mask))
            self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)

        cost = tf.losses.huber_loss(
            target, pred_action_value, reduction=tf.losses.Reduction.MEAN)
        # summary.add_param_summary(('conv.*/W', ['histogram', 'rms']),
        #                           ('fc.*/W', ['histogram', 'rms']))   # monitor all W
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.learning_rate, trainable=False)
        # opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(
            opt, [gradproc.MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5)),
                  gradproc.SummaryGradient()])

    @staticmethod
    def update_target_param():
        vars = tf.global_variables()
        ops = []
        G = tf.get_default_graph()
        for v in vars:
            target_name = v.op.name
            if target_name.startswith('target'):
                new_name = target_name.replace('target/', '')
                logger.info("Target Network Update: {} <- {}".format(target_name, new_name))
                ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network')