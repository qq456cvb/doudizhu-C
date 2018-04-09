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
from env import Env as CEnv
from utils import get_seq_length, pick_minor_targets, to_char, discard_onehot_from_s_60
from utils import pick_main_cards
from six.moves import queue

from pyenv import Pyenv
import tensorflow.contrib.slim as slim
from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import get_current_tower_context, optimizer

from TensorPack.A3C.simulator import SimulatorProcess, SimulatorMaster, TransitionExperience
from TensorPack.A3C.model_loader import ModelLoader
from TensorPack.A3C.evaluator import Evaluator
from TensorPack.PolicySL.Policy_SL_v1_4 import conv_block as policy_conv_block
from TensorPack.ValueSL.Value_SL_v1_4 import conv_block as value_conv_block

import six
import numpy as np

if six.PY3:
    from concurrent import futures

    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

GAMMA = 0.99
POLICY_INPUT_DIM = 60 * 3
POLICY_LAST_INPUT_DIM = 60
POLICY_WEIGHT_DECAY = 5 * 1e-4
VALUE_INPUT_DIM = 60 * 3
LORD_ID = 2
SIMULATOR_PROC = 40

# number of games per epoch roughly = STEPS_PER_EPOCH * BATCH_SIZE / 100
STEPS_PER_EPOCH = 100
BATCH_SIZE = 1024
PREDICT_BATCH_SIZE = 64
PREDICTOR_THREAD_PER_GPU = 1
PREDICTOR_THREAD = None


def get_player():
    return CEnv()


class Model(ModelDesc):
    def get_policy(self, role_id, state, last_cards, minor_type):
        # policy network, different for three agents
        batch_size = tf.shape(role_id)[0]
        gathered_outputs = []
        indices = []
        for idx in range(1, 4):
            with tf.variable_scope('policy_network_%d' % idx):
                id_idx = tf.where(tf.equal(role_id, idx))
                indices.append(id_idx)
                state_id = tf.gather(state, id_idx)
                last_cards_id = tf.gather(last_cards, id_idx)
                minor_type_id = tf.gather(minor_type, id_idx)
                with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(POLICY_WEIGHT_DECAY)):
                    with tf.variable_scope('branch_main'):
                        flattened_1 = policy_conv_block(state_id[:, :60], 32, POLICY_INPUT_DIM // 3,
                                                        [[16, 32, 5, 'identity'],
                                                         [16, 32, 5, 'identity'],
                                                         [32, 128, 5, 'upsampling'],
                                                         [32, 128, 5, 'identity'],
                                                         [32, 128, 5, 'identity'],
                                                         [64, 256, 5, 'upsampling'],
                                                         [64, 256, 3, 'identity'],
                                                         [64, 256, 3, 'identity']
                                                         ], 'branch_main1')
                        flattened_2 = policy_conv_block(state_id[:, 60:120], 32, POLICY_INPUT_DIM // 3,
                                                        [[16, 32, 5, 'identity'],
                                                         [16, 32, 5, 'identity'],
                                                         [32, 128, 5, 'upsampling'],
                                                         [32, 128, 5, 'identity'],
                                                         [32, 128, 5, 'identity'],
                                                         [64, 256, 5, 'upsampling'],
                                                         [64, 256, 3, 'identity'],
                                                         [64, 256, 3, 'identity']
                                                         ], 'branch_main2')
                        flattened_3 = policy_conv_block(state_id[:, 120:], 32, POLICY_INPUT_DIM // 3,
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
                        flattened_last = policy_conv_block(last_cards_id, 32, POLICY_LAST_INPUT_DIM,
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
                        minor_type_embedding = slim.fully_connected(inputs=tf.one_hot(minor_type_id, 2), num_outputs=256,
                                                                    activation_fn=tf.nn.sigmoid)
                        fc_minor = fc_minor * minor_type_embedding

                        fc_minor = slim.fully_connected(inputs=fc_minor, num_outputs=64, activation_fn=tf.nn.relu)
                        minor_response_logits = slim.fully_connected(inputs=fc_minor, num_outputs=15,
                                                                     activation_fn=None)

            gathered_outputs.append([passive_decision_logits, passive_bomb_logits, passive_response_logits,
                   active_decision_logits, active_response_logits, active_seq_logits, minor_response_logits])

        # 7: B * ?
        outputs = []
        for i in range(7):
            scatter_shape = tf.cast(tf.stack([batch_size, tf.shape([gathered_outputs[0][i]])[1]]), dtype=tf.int64)
            outputs.append(tf.add_n([tf.scatter_nd(indices[k], gathered_outputs[k][i], scatter_shape) for k in range(3)]))

        return outputs

    def get_value(self, role_id, state):
        with tf.variable_scope('value_network'):
            # not adding regular loss for fc since we need big scalar output [-1, 1]
            with tf.variable_scope('value_conv'):
                flattened_1 = value_conv_block(state[:, :60], 32, VALUE_INPUT_DIM // 3, [[16, 32, 5, 'identity'],
                                                                  [16, 32, 5, 'identity'],
                                                                  [32, 128, 5, 'upsampling'],
                                                                  [32, 128, 3, 'identity'],
                                                                  [32, 128, 3, 'identity'],
                                                                  [64, 256, 3, 'upsampling'],
                                                                  [64, 256, 3, 'identity'],
                                                                  [64, 256, 3, 'identity']
                                                                  ], 'value_conv1')
                flattened_2 = value_conv_block(state[:, 60:120], 32, VALUE_INPUT_DIM // 3, [[16, 32, 5, 'identity'],
                                                                  [16, 32, 5, 'identity'],
                                                                  [32, 128, 5, 'upsampling'],
                                                                  [32, 128, 3, 'identity'],
                                                                  [32, 128, 3, 'identity'],
                                                                  [64, 256, 3, 'upsampling'],
                                                                  [64, 256, 3, 'identity'],
                                                                  [64, 256, 3, 'identity']
                                                                  ], 'value_conv2')
                flattened_3 = value_conv_block(state[:, 120:], 32, VALUE_INPUT_DIM // 3, [[16, 32, 5, 'identity'],
                                                                  [16, 32, 5, 'identity'],
                                                                  [32, 128, 5, 'upsampling'],
                                                                  [32, 128, 3, 'identity'],
                                                                  [32, 128, 3, 'identity'],
                                                                  [64, 256, 3, 'upsampling'],
                                                                  [64, 256, 3, 'identity'],
                                                                  [64, 256, 3, 'identity']
                                                                  ], 'value_conv3')
                flattened = tf.concat([flattened_1, flattened_2, flattened_3], axis=1)

            with tf.variable_scope('value_fc'):
                value = slim.fully_connected(flattened, num_outputs=1, activation_fn=None)

        value = tf.squeeze(value, 1)
        indicator = tf.cast(tf.equal(role_id, LORD_ID), tf.float32) * 2 - 1
        return -value * indicator

    def inputs(self):
        return [
            tf.placeholder(tf.int32, [None], 'role_id'),
            tf.placeholder(tf.float32, [None, POLICY_INPUT_DIM], 'policy_state_in'),
            tf.placeholder(tf.float32, [None, VALUE_INPUT_DIM], 'value_state_in'),
            tf.placeholder(tf.float32, [None, POLICY_LAST_INPUT_DIM], 'last_cards_in'),
            tf.placeholder(tf.int32, [None], 'passive_decision_in'),
            tf.placeholder(tf.int32, [None], 'passive_bomb_in'),
            tf.placeholder(tf.int32, [None], 'passive_response_in'),
            tf.placeholder(tf.int32, [None], 'active_decision_in'),
            tf.placeholder(tf.int32, [None], 'active_response_in'),
            tf.placeholder(tf.int32, [None], 'sequence_length_in'),
            tf.placeholder(tf.int32, [None], 'minor_response_in'),
            tf.placeholder(tf.int32, [None], 'minor_type_in'),
            tf.placeholder(tf.int32, [None], 'mode_in'),
            tf.placeholder(tf.float32, [None], 'history_action_prob_in'),
            tf.placeholder(tf.float32, [None], 'discounted_return_in')
        ]

    def build_graph(self, role_id, prob_state, value_state, last_cards, passive_decision_target, passive_bomb_target, passive_response_target,
                    active_decision_target, active_response_target, seq_length_target, minor_response_target,
                    minor_type, mode, history_action_prob, discounted_return):

        (passive_decision_logits, passive_bomb_logits, passive_response_logits, active_decision_logits,
         active_response_logits, active_seq_logits, minor_response_logits) = self.get_policy(role_id, prob_state, last_cards,
                                                                                           minor_type)
        passive_decision_prob = tf.nn.softmax(passive_decision_logits, name='passive_decision_prob')
        passive_bomb_prob = tf.nn.softmax(passive_bomb_logits, name='passive_bomb_prob')
        passive_response_prob = tf.nn.softmax(passive_response_logits, name='passive_response_prob')
        active_decision_prob = tf.nn.softmax(active_decision_logits, name='active_decision_prob')
        active_response_prob = tf.nn.softmax(active_response_logits, name='active_response_prob')
        active_seq_prob = tf.nn.softmax(active_seq_logits, name='active_seq_prob')
        minor_response_prob = tf.nn.softmax(minor_response_logits, name='minor_response_prob')
        mode_out = tf.identity(mode, name='mode_out')
        value = self.get_value(role_id, value_state)
        # this is the value for each agent, not the global value
        value = tf.identity(value, name='pred_value')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return

        # passive mode
        passive_decision_logpa = tf.reduce_sum(tf.one_hot(passive_decision_target, 4) * tf.log(
            tf.clip_by_value(passive_decision_prob, 1e-7, 1 - 1e-7)), 1)

        passive_response_logpa = tf.reduce_sum(tf.one_hot(passive_response_target, 15) * tf.log(
            tf.clip_by_value(passive_response_prob, 1e-7, 1 - 1e-7)), 1)

        passive_bomb_logpa = tf.reduce_sum(tf.one_hot(passive_bomb_target, 13) * tf.log(
            tf.clip_by_value(passive_bomb_prob, 1e-7, 1 - 1e-7)), 1)

        # active mode
        active_decision_logpa = tf.reduce_sum(tf.one_hot(active_decision_target, 13) * tf.log(
            tf.clip_by_value(active_decision_prob, 1e-7, 1 - 1e-7)), 1)

        active_response_logpa = tf.reduce_sum(tf.one_hot(active_response_target, 15) * tf.log(
            tf.clip_by_value(active_response_prob, 1e-7, 1 - 1e-7)), 1)

        active_seq_logpa = tf.reduce_sum(tf.one_hot(seq_length_target, 12) * tf.log(
            tf.clip_by_value(active_seq_prob, 1e-7, 1 - 1e-7)), 1)

        # minor mode
        minor_response_logpa = tf.reduce_sum(tf.one_hot(minor_response_target, 15) * tf.log(
            tf.clip_by_value(minor_response_prob, 1e-7, 1 - 1e-7)), 1)

        # B * 7
        logpa = tf.stack([passive_decision_logpa, passive_response_logpa, passive_bomb_logpa, active_decision_logpa, active_response_logpa, active_seq_logpa, minor_response_logpa], axis=1)
        idx = tf.stack([tf.range(tf.shape(prob_state)[0]), mode], axis=1)

        # B
        logpa = tf.gather_nd(logpa, idx)

        # importance sampling
        passive_decision_pa = tf.reduce_sum(tf.one_hot(passive_decision_target, 4) * tf.clip_by_value(passive_decision_prob, 1e-7, 1 - 1e-7), 1)
        passive_response_pa = tf.reduce_sum(tf.one_hot(passive_response_target, 15) * tf.clip_by_value(passive_response_prob, 1e-7, 1 - 1e-7), 1)
        passive_bomb_pa = tf.reduce_sum(tf.one_hot(passive_bomb_target, 13) * tf.clip_by_value(passive_bomb_prob, 1e-7, 1 - 1e-7), 1)
        active_decision_pa = tf.reduce_sum(tf.one_hot(active_decision_target, 13) * tf.clip_by_value(active_decision_prob, 1e-7, 1 - 1e-7), 1)
        active_response_pa = tf.reduce_sum(tf.one_hot(active_response_target, 15) * tf.clip_by_value(active_response_prob, 1e-7, 1 - 1e-7), 1)
        active_seq_pa = tf.reduce_sum(tf.one_hot(seq_length_target, 12) * tf.clip_by_value(active_seq_prob, 1e-7, 1 - 1e-7), 1)
        minor_response_pa = tf.reduce_sum(tf.one_hot(minor_response_target, 15) * tf.clip_by_value(minor_response_prob, 1e-7, 1 - 1e-7), 1)

        # B * 7
        pa = tf.stack([passive_decision_pa, passive_response_pa, passive_bomb_pa, active_decision_pa, active_response_pa, active_seq_pa, minor_response_pa], axis=1)
        idx = tf.stack([tf.range(tf.shape(prob_state)[0]), mode], axis=1)

        # B
        pa = tf.gather_nd(pa, idx)
        importance_b = tf.stop_gradient(tf.clip_by_value(pa / (history_action_prob + 1e-8), 0, 10))

        # advantage
        advantage_b = tf.subtract(discounted_return, tf.stop_gradient(value), name='advantage')

        policy_loss_b = -logpa * advantage_b * importance_b
        entropy_loss_b = pa * logpa
        value_loss_b = tf.square(value - discounted_return)

        entropy_beta = tf.get_variable('entropy_beta', shape=[], initializer=tf.constant_initializer(0.01),
                                       trainable=False)
        # print(policy_loss_b.shape)
        # print(entropy_loss_b.shape)
        # print(value_loss_b.shape)
        # print(advantage_b.shape)
        costs = []
        for i in range(1, 4):
            mask = tf.equal(role_id, i)
            # print(mask.shape)
            pred_reward = tf.reduce_mean(tf.boolean_mask(value, mask), name='predict_reward_%d' % i)
            advantage = tf.sqrt(tf.reduce_mean(tf.square(tf.boolean_mask(advantage_b, mask))), name='rms_advantage_%d' % i)

            policy_loss = tf.reduce_sum(tf.boolean_mask(policy_loss_b, mask, name='policy_loss_%d' % i))
            entropy_loss = tf.reduce_sum(tf.boolean_mask(entropy_loss_b, mask, name='entropy_loss_%d' % i))
            value_loss = tf.reduce_sum(tf.boolean_mask(value_loss_b, mask, name='value_loss_%d' % i))
            cost = tf.add_n([policy_loss, entropy_loss * entropy_beta, value_loss])
            cost = tf.truediv(cost, tf.reduce_sum(tf.cast(mask, tf.float32)), name='cost_%d' % i)
            costs.append(cost)

            importance = tf.reduce_mean(tf.boolean_mask(importance_b, mask), name='importance_%d' % i)
            add_moving_summary(policy_loss, entropy_loss, value_loss, pred_reward, advantage, cost, importance, decay=0.1)

        return tf.add_n(costs)

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.3))]
                     # SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return CEnv()


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, gpus):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)
        self._gpus = gpus

    def _setup_graph(self):
        # create predictors on the available predictor GPUs.
        nr_gpu = len(self._gpus)
        predictors = [self.trainer.get_predictor(
            ['role_id', 'policy_state_in', 'value_state_in', 'last_cards_in', 'minor_type_in', 'mode_in'],
            ['passive_decision_prob', 'passive_bomb_prob', 'passive_response_prob', 'active_decision_prob',
             'active_response_prob', 'active_seq_prob', 'minor_response_prob', 'mode_out', 'pred_value'],
            self._gpus[k % nr_gpu])
            for k in range(PREDICTOR_THREAD)]
        self.async_predictor = MultiThreadAsyncPredictor(
            predictors, batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()

    def _on_state(self, role_id, prob_state, all_state, last_cards_onehot, mask, minor_type, mode, first_st, client):
        """
        Launch forward prediction for the new state given by some client.
        """
        def cb(outputs):
            try:
                output = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(client.ident))
                return
            value = output[-1]
            mode = output[-2]
            distrib = (output[:-2][mode] + 1e-6) * mask
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib / distrib.sum())
            client.memory[role_id - 1].append(TransitionExperience(
                prob_state, all_state, action, reward=0, minor_type=minor_type, first_st=first_st,
                last_cards_onehot=last_cards_onehot, mode=mode, value=value, prob=distrib[action]))
            self.send_queue.put([client.ident, dumps(action)])
        self.async_predictor.put_task([role_id, prob_state, all_state, last_cards_onehot, minor_type, mode], cb)

    def _process_msg(self, client, role_id, prob_state, all_state, last_cards_onehot, first_st, mask, minor_type, mode, reward, isOver):
        """
        Process a message sent from some client.
        """
        # in the first message, only state is valid,
        # reward&isOver should be discarde
        if isOver and first_st:
            # should clear client's memory and put to queue
            assert reward != 0
            for i in range(3):
                j = -1
                while client.memory[i][j].reward == 0:
                    # notice that C++ returns the reward for farmer, transform to the reward in each agent's perspective
                    client.memory[i][j].reward = reward if i != 1 else -reward
                    if client.memory[i][j].first_st:
                        break
                    j -= 1
            self._parse_memory(0, client)
        # feed state and return action
        self._on_state(role_id, prob_state, all_state, last_cards_onehot, mask, minor_type, mode, first_st, client)

    def _parse_memory(self, init_r, client):
        # for each agent's memory
        for role_id in range(1, 4):
            mem = client.memory[role_id - 1]

            mem.reverse()
            R = float(init_r)
            mem_valid = [m for m in mem if m.first_st]
            dr = []
            for idx, k in enumerate(mem_valid):
                R = np.clip(k.reward, -1, 1) + GAMMA * R
                dr.append(R)
            dr.reverse()
            mem.reverse()
            i = -1
            j = 0
            while j < len(mem):
                if mem[j].first_st:
                    i += 1
                target = [0 for _ in range(7)]
                k = mem[j]
                target[k.mode] = k.action
                # print('pushed to queue')
                # sys.stdout.flush()
                self.queue.put([role_id, k.prob_state, k.all_state, k.last_cards_onehot, *target, k.minor_type, k.mode, k.prob, dr[i]])
                j += 1

            client.memory[role_id - 1] = []


def train():
    dirname = os.path.join('train_log', 'a3c')
    logger.set_logger_dir(dirname)

    # assign GPUs for training & inference
    nr_gpu = get_nr_gpu()
    global PREDICTOR_THREAD
    if nr_gpu > 0:
        if nr_gpu > 1:
            # use all gpus for inference
            predict_tower = list(range(nr_gpu))
        else:
            predict_tower = [0]
        PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
        train_tower = list(range(nr_gpu))[:-nr_gpu // 2] or [0]
        logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
            ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
    else:
        logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
        PREDICTOR_THREAD = 1
        predict_tower, train_tower = [0], [0]

    # setup simulator processes
    name_base = str(uuid.uuid1())[:6]
    prefix = '@' if sys.platform.startswith('linux') else ''
    namec2s = 'ipc://{}sim-c2s-{}'.format(prefix, name_base)
    names2c = 'ipc://{}sim-s2c-{}'.format(prefix, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, predict_tower)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)
    config = AutoResumeTrainConfig(
        model=Model(),
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            # ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            # ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            master,
            StartProcOrThread(master),
            Evaluator(
                100, ['role_id', 'policy_state_in', 'last_cards_in', 'minor_type_in'],
                ['passive_decision_prob', 'passive_bomb_prob', 'passive_response_prob',
                 'active_decision_prob', 'active_response_prob', 'active_seq_prob', 'minor_response_prob'], get_player),
        ],
        session_init=ModelLoader('policy_network_2', 'SL_policy_network', 'value_network', 'SL_value_network'),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )
    trainer = SimpleTrainer() if config.nr_tower == 1 else AsyncMultiGPUTrainer(train_tower)
    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    train()
