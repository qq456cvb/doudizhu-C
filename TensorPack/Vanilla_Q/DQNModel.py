#!/usr/bin/env python
# -*- coding: utf-8 -*-`
# File: DQNModel.py


import abc
import tensorflow as tf
import tensorpack
from tensorpack import ModelDesc
from tensorpack.utils import logger
from tensorpack.tfutils import (
    varreplace, summary, get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import numpy as np
from tensorflow.contrib.layers import l2_regularizer
import os


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
                               (None, 2, self.state_shape[0]),
                               'joint_state'),
                tf.placeholder(tf.bool, (None, self.num_actions), 'next_mask'),
                tf.placeholder(tf.int64, (None,), 'action'),
                tf.placeholder(tf.float32, (None,), 'reward'),
                tf.placeholder(tf.bool, (None,), 'isOver')]

    # input B * C * D
    # output B * C * 1
    @abc.abstractmethod
    def _get_DQN_prediction(self, state):
        pass

    # decorate the function
    # output : B * A
    @auto_reuse_variable_scope
    def get_DQN_prediction(self, state):
        state_embeddings = self.get_state_embedding(state)
        encoding = np.load(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '../AutoEncoder/encoding.npy'))
        actions = tf.convert_to_tensor(encoding)
        action_embeddings = self.get_action_embedding(actions)
        return tf.identity(tf.reduce_sum(tf.expand_dims(state_embeddings, 1) * tf.expand_dims(action_embeddings, 0), -1), 'Qvalue')

    @auto_reuse_variable_scope
    def get_state_embedding(self, state):
        pass

    @auto_reuse_variable_scope
    def get_action_embedding(self, action):
        pass

    # joint state: B * 2 * COMB * N * D for now, D = 256
    # dynamic action range
    def build_graph(self, joint_state, next_mask, action, reward, isOver):
        state = tf.identity(joint_state[:, 0, ...], name='state')
        self.predict_value = self.get_DQN_prediction(state)
        if not get_current_tower_context().is_training:
            return

        next_state = tf.identity(joint_state[:, 1, ...], name='next_state')
        action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)

        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        summary.add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'), varreplace.freeze_variables(skip_collection=True):
            # we are alternating between comb and fine states
            targetQ_predict_value = self.get_DQN_prediction(next_state)    # NxA

        if self.method != 'Double':
            # DQN
            self.greedy_choice = tf.argmax(targetQ_predict_value + (tf.to_float(next_mask) * 1e4), 1)  # N,
            predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)
        else:
            # Double-DQN
            next_predict_value = self.get_DQN_prediction(next_state)
            self.greedy_choice = tf.argmax(next_predict_value + (tf.to_float(next_mask) * 1e4), 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)

        l2_loss = tensorpack.regularize_cost('.*W{1}', l2_regularizer(1e-3))
        # cost = tf.losses.mean_squared_error(target, pred_action_value)
        cost = tf.losses.huber_loss(
                        target, pred_action_value, reduction=tf.losses.Reduction.MEAN)
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))   # monitor all W
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.learning_rate, trainable=False)
        # opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(
            opt, [
                # gradproc.GlobalNormClip(2.0),
                gradproc.MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5)),
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
