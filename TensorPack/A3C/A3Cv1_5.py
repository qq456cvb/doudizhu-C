import numpy as np
import os
import uuid
import argparse

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import sys
import os

if os.name == 'nt':
    sys.path.insert(0, '../../build/Release')
else:
    sys.path.insert(0, '../../build.linux')
from env import Env
from utils import get_seq_length, pick_minor_targets, to_char, discard_onehot_from_s_60
from utils import pick_main_cards

import tensorflow.contrib.slim as slim
from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_nr_gpu

from TensorPack.A3C.simulator import SimulatorProcess, SimulatorMaster, TransitionExperience
from TensorPack.PolicySL.Policy_SL_v1_4 import conv_block as policy_conv_block

import six

if six.PY3:
    from concurrent import futures

    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

GAMMA = 0.99
POLICY_INPUT_DIM = 60 * 3
POLICY_LAST_INPUT_DIM = 60
POLICY_WEIGHT_DECAY = 5 * 1e-4

# number of games per epoch roughly = STEPS_PER_EPOCH * BATCH_SIZE / 100
STEPS_PER_EPOCH = 1000
BATCH_SIZE = 1024
PREDICT_BATCH_SIZE = 128
PREDICTOR_THREAD_PER_GPU = 3
PREDICTOR_THREAD = None


class Model(ModelDesc):
    def get_pred(self, role_id, state, last_cards, minor_type):
        # policy network, different for three agents
        for idx in range(1, 4):
            with tf.variable_scope('policy_network_' % idx):
                with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(POLICY_WEIGHT_DECAY)):
                    with tf.variable_scope('branch_main'):
                        flattened_1 = policy_conv_block(state[:, :60], 32, POLICY_INPUT_DIM // 3,
                                                        [[16, 32, 5, 'identity'],
                                                         [16, 32, 5, 'identity'],
                                                         [32, 128, 5, 'upsampling'],
                                                         [32, 128, 5, 'identity'],
                                                         [32, 128, 5, 'identity'],
                                                         [64, 256, 5, 'upsampling'],
                                                         [64, 256, 3, 'identity'],
                                                         [64, 256, 3, 'identity']
                                                         ], 'branch_main1')
                        flattened_2 = policy_conv_block(state[:, 60:120], 32, POLICY_INPUT_DIM // 3,
                                                        [[16, 32, 5, 'identity'],
                                                         [16, 32, 5, 'identity'],
                                                         [32, 128, 5, 'upsampling'],
                                                         [32, 128, 5, 'identity'],
                                                         [32, 128, 5, 'identity'],
                                                         [64, 256, 5, 'upsampling'],
                                                         [64, 256, 3, 'identity'],
                                                         [64, 256, 3, 'identity']
                                                         ], 'branch_main2')
                        flattened_3 = policy_conv_block(state[:, 120:], 32, POLICY_INPUT_DIM // 3,
                                                        [[16, 32, 5, 'identity'],
                                                         [16, 32, 5, 'identity'],
                                                         [32, 128, 5, 'upsampling'],
                                                         [32, 128, 5, 'identity'],
                                                         [32, 128, 5, 'identity'],
                                                         [64, 256, 5, 'upsampling'],
                                                         [64, 256, 3, 'identity'],
                                                         [64, 256, 3, 'identity']
                                                         ], 'branch_main3')
                        flattened = tf.concat([flattened_1, flattened_2, flattened_3], axis=1)

                    with tf.variable_scope('branch_passive'):
                        flattened_last = policy_conv_block(last_cards, 32, POLICY_LAST_INPUT_DIM,
                                                           [[16, 32, 5, 'identity'],
                                                            [16, 32, 5, 'identity'],
                                                            [32, 128, 5, 'upsampling'],
                                                            [32, 128, 5, 'identity'],
                                                            [32, 128, 5, 'identity'],
                                                            [64, 256, 5, 'upsampling'],
                                                            [64, 256, 3, 'identity'],
                                                            [64, 256, 3, 'identity']
                                                            ], 'last_cards')

                        # no regularization for LSTM yet
                        with tf.variable_scope('decision'):
                            attention_decision = slim.fully_connected(inputs=flattened_last, num_outputs=256,
                                                                      activation_fn=tf.nn.sigmoid)

                            fc_passive_decision = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                       activation_fn=tf.nn.relu)
                            fc_passive_decision = fc_passive_decision * attention_decision
                            fc_passive_decision = slim.fully_connected(inputs=fc_passive_decision, num_outputs=64,
                                                                       activation_fn=tf.nn.relu)
                            passive_decision_logits = slim.fully_connected(inputs=fc_passive_decision,
                                                                           num_outputs=4,
                                                                           activation_fn=None)

                        # bomb and response do not depend on each other
                        with tf.variable_scope('bomb'):
                            fc_passive_bomb = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                   activation_fn=tf.nn.relu)
                            fc_passive_bomb = slim.fully_connected(inputs=fc_passive_bomb, num_outputs=64,
                                                                   activation_fn=tf.nn.relu)
                            passive_bomb_logits = slim.fully_connected(inputs=fc_passive_bomb, num_outputs=13,
                                                                       activation_fn=None)

                        with tf.variable_scope('response'):
                            attention_response = slim.fully_connected(inputs=flattened_last, num_outputs=256,
                                                                      activation_fn=tf.nn.sigmoid)

                            fc_passive_response = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                       activation_fn=tf.nn.relu)
                            fc_passive_response = fc_passive_response * attention_response
                            fc_passive_response = slim.fully_connected(inputs=fc_passive_response, num_outputs=64,
                                                                       activation_fn=tf.nn.relu)
                            passive_response_logits = slim.fully_connected(inputs=fc_passive_response,
                                                                           num_outputs=15,
                                                                           activation_fn=None)

                    with tf.variable_scope('branch_active'):
                        hidden_size = 256
                        lstm_active = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

                        with tf.variable_scope('decision'):
                            fc_active_decision = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                      activation_fn=tf.nn.relu)
                            lstm_active_decision_output, hidden_active_output = tf.nn.dynamic_rnn(lstm_active,
                                                                                                  tf.expand_dims(
                                                                                                      fc_active_decision,
                                                                                                      1),
                                                                                                  initial_state=lstm_active.zero_state(
                                                                                                      tf.shape(
                                                                                                          fc_active_decision)[
                                                                                                          0],
                                                                                                      dtype=tf.float32),
                                                                                                  sequence_length=tf.ones(
                                                                                                      [
                                                                                                          tf.shape(
                                                                                                              state)[
                                                                                                              0]]))
                            fc_active_decision = slim.fully_connected(
                                inputs=tf.squeeze(lstm_active_decision_output, axis=[1]), num_outputs=64,
                                activation_fn=tf.nn.relu)
                            active_decision_logits = slim.fully_connected(inputs=fc_active_decision, num_outputs=13,
                                                                          activation_fn=None)

                        with tf.variable_scope('response'):
                            fc_active_response = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                      activation_fn=tf.nn.relu)
                            lstm_active_response_output, hidden_active_output = tf.nn.dynamic_rnn(lstm_active,
                                                                                                  tf.expand_dims(
                                                                                                      fc_active_response,
                                                                                                      1),
                                                                                                  initial_state=hidden_active_output,
                                                                                                  sequence_length=tf.ones(
                                                                                                      [
                                                                                                          tf.shape(
                                                                                                              state)[
                                                                                                              0]]))
                            fc_active_response = slim.fully_connected(
                                inputs=tf.squeeze(lstm_active_response_output, axis=[1]), num_outputs=64,
                                activation_fn=tf.nn.relu)
                            active_response_logits = slim.fully_connected(inputs=fc_active_response, num_outputs=15,
                                                                          activation_fn=None)

                        with tf.variable_scope('seq_length'):
                            fc_active_seq = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                 activation_fn=tf.nn.relu)
                            lstm_active_seq_output, _ = tf.nn.dynamic_rnn(lstm_active,
                                                                          tf.expand_dims(fc_active_seq, 1),
                                                                          initial_state=hidden_active_output,
                                                                          sequence_length=tf.ones(
                                                                              [tf.shape(state)[0]]))
                            fc_active_seq = slim.fully_connected(inputs=tf.squeeze(lstm_active_seq_output, axis=[1]),
                                                                 num_outputs=64, activation_fn=tf.nn.relu)
                            active_seq_logits = slim.fully_connected(inputs=fc_active_seq, num_outputs=12,
                                                                     activation_fn=None)

                    with tf.variable_scope('branch_minor'):
                        fc_minor = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                        activation_fn=tf.nn.relu)
                        minor_type_embedding = slim.fully_connected(inputs=tf.one_hot(minor_type, 2), num_outputs=256,
                                                                    activation_fn=tf.nn.sigmoid)
                        fc_minor = fc_minor * minor_type_embedding

                        fc_minor = slim.fully_connected(inputs=fc_minor, num_outputs=64, activation_fn=tf.nn.relu)
                        minor_response_logits = slim.fully_connected(inputs=fc_minor, num_outputs=15,
                                                                     activation_fn=None)


            return passive_decision_logits, passive_bomb_logits, passive_response_logits, \
                   active_decision_logits, active_response_logits, active_seq_logits, minor_response_logits

    def inputs(self):
        return [
            tf.placeholder(tf.int32, [None], 'role_id'),
            tf.placeholder(tf.float32, [None, POLICY_INPUT_DIM], 'state_in'),
            tf.placeholder(tf.float32, [None, POLICY_LAST_INPUT_DIM], 'last_cards_in'),
            tf.placeholder(tf.int32, [None], 'passive_decision_in'),
            tf.placeholder(tf.int32, [None], 'passive_bomb_in'),
            tf.placeholder(tf.int32, [None], 'passive_response_in'),
            tf.placeholder(tf.int32, [None], 'active_decision_in'),
            tf.placeholder(tf.int32, [None], 'active_response_in'),
            tf.placeholder(tf.int32, [None], 'sequence_length_in'),
            tf.placeholder(tf.int32, [None], 'minor_response_in'),
            tf.placeholder(tf.int32, [None], 'minor_type_in'),
            tf.placeholder(tf.int32, [None], 'mode_in')
        ]

    def build_graph(self, role_id, state, last_cards, passive_decision_target, passive_bomb_target, passive_response_target,
                    active_decision_target, active_response_target, seq_length_target, minor_response_target,
                    minor_type, mode):
        (passive_decision_logits, passive_bomb_logits, passive_response_logits, active_decision_logits,
         active_response_logits, active_seq_logits, minor_response_logits) = self.get_pred(role_id, state, last_cards,
                                                                                           minor_type)
        passive_decision_prob = tf.nn.softmax(passive_decision_logits, name='passive_decision_prob')
        passive_bomb_prob = tf.nn.softmax(passive_bomb_logits, name='passive_bomb_prob')
        passive_response_prob = tf.nn.softmax(passive_response_logits, name='passive_response_prob')
        active_decision_prob = tf.nn.softmax(active_decision_logits, name='active_decision_prob')
        active_response_prob = tf.nn.softmax(active_response_logits, name='active_response_prob')
        active_seq_prob = tf.nn.softmax(active_seq_logits, name='active_seq_prob')
        minor_response_prob = tf.nn.softmax(minor_response_logits, name='minor_response_prob')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return

        # passive mode
        with tf.variable_scope("passive_mode_loss"):
            passive_decision_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.one_hot(passive_decision_target, 4), logits=passive_decision_logits)
            passive_bomb_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(passive_bomb_target, 13),
                                                                           logits=passive_bomb_logits)
            passive_response_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.one_hot(passive_response_target, 15), logits=passive_response_logits)

        # active mode
        with tf.variable_scope("active_mode_loss"):
            active_decision_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.one_hot(active_decision_target, 13), logits=active_decision_logits)
            active_response_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.one_hot(active_response_target, 15), logits=active_response_logits)
            active_seq_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(seq_length_target, 12),
                                                                         logits=active_seq_logits)

        with tf.variable_scope("minor_mode_loss"):
            minor_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(minor_response_target, 15),
                                                                    logits=minor_response_logits)

        # B * 7
        losses = [passive_decision_loss, passive_bomb_loss, passive_response_loss,
                  active_decision_loss, active_response_loss, active_seq_loss, minor_loss]

        losses = tf.stack(losses, axis=1)
        idx = tf.stack([tf.range(0, tf.shape(state)[0]), mode], axis=1)
        loss = tf.gather_nd(losses, idx)
        print(loss.shape)
        loss = tf.reduce_mean(loss, name='loss')

        add_moving_summary(loss, decay=0.1)
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.3)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt
