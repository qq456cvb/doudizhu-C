import numpy as np
import os
import uuid
import argparse

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

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

from TensorPack.A3C_FC.simulator_fc import SimulatorProcess, SimulatorMaster, TransitionExperience
from TensorPack.A3C.model_loader import ModelLoader
from TensorPack.A3C_FC.evaluator_fc import Evaluator
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
POLICY_INPUT_DIM = 9085
POLICY_LAST_INPUT_DIM = 9085
POLICY_WEIGHT_DECAY = 1e-3
VALUE_INPUT_DIM = 9085 * 3
LORD_ID = 2
SIMULATOR_PROC = 50

# number of games per epoch roughly = STEPS_PER_EPOCH * BATCH_SIZE / 100
STEPS_PER_EPOCH = 1000
BATCH_SIZE = 64
PREDICT_BATCH_SIZE = 32
PREDICTOR_THREAD_PER_GPU = 2
PREDICTOR_THREAD = None


def get_player():
    return CEnv()


def res_fc_block(inputs, units, stack=3):
    residual = inputs
    for _ in range(stack):
        residual = slim.fully_connected(residual, units)
    x = inputs
    if inputs.shape[1].value != units:
        x = slim.fully_connected(x, units)
    return tf.contrib.layers.layer_norm(residual + x, scale=False)


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
                state_id = tf.gather_nd(state, id_idx)
                last_cards_id = tf.gather_nd(last_cards, id_idx)
                minor_type_id = tf.gather_nd(minor_type, id_idx)
                with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(POLICY_WEIGHT_DECAY)):
                    with tf.variable_scope('branch_main'):
                        x = state_id
                        feats = [1024, 512, 512, 256, 256]
                        for f in feats:
                            for _ in range(3):
                                x = res_fc_block(x, f)
                        flattened = x

                    with tf.variable_scope('branch_passive'):
                        x = last_cards_id
                        for f in feats:
                            for _ in range(3):
                                x = res_fc_block(x, f)
                        flattened_last = x

                        # no regularization for LSTM yet
                        with tf.variable_scope('decision'):
                            attention_decision = slim.fully_connected(inputs=res_fc_block(flattened_last, 256), num_outputs=256,
                                                                      activation_fn=tf.nn.sigmoid)

                            fc_passive_decision = res_fc_block(flattened, 256)
                            fc_passive_decision = fc_passive_decision * attention_decision
                            fc_passive_decision = res_fc_block(fc_passive_decision, 64)
                            passive_decision_logits = slim.fully_connected(inputs=res_fc_block(fc_passive_decision, 64),
                                                                           num_outputs=4,
                                                                           activation_fn=None)

                        # bomb and response do not depend on each other
                        with tf.variable_scope('bomb'):
                            fc_passive_bomb = res_fc_block(flattened, 256)
                            fc_passive_bomb = res_fc_block(fc_passive_bomb, 64)
                            passive_bomb_logits = slim.fully_connected(inputs=res_fc_block(fc_passive_bomb, 64), num_outputs=13,
                                                                       activation_fn=None)

                        with tf.variable_scope('response'):
                            attention_response = slim.fully_connected(inputs=res_fc_block(flattened_last, 256), num_outputs=256,
                                                                      activation_fn=tf.nn.sigmoid)

                            fc_passive_response = res_fc_block(flattened, 256)
                            fc_passive_response = fc_passive_response * attention_response
                            fc_passive_response = res_fc_block(fc_passive_response, 64)
                            passive_response_logits = slim.fully_connected(inputs=res_fc_block(fc_passive_response, 64),
                                                                           num_outputs=15,
                                                                           activation_fn=None)

                    with tf.variable_scope('branch_active'):
                        hidden_size = 256
                        lstm_active = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

                        with tf.variable_scope('decision'):
                            fc_active_decision = res_fc_block(flattened, 256)
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
                                                                                                              state_id)[
                                                                                                              0]]))
                            fc_active_decision = res_fc_block(tf.squeeze(lstm_active_decision_output, axis=[1]), 64)
                            active_decision_logits = slim.fully_connected(inputs=res_fc_block(fc_active_decision, 64), num_outputs=13,
                                                                          activation_fn=None)

                        with tf.variable_scope('response'):
                            fc_active_response = res_fc_block(flattened, 256)
                            lstm_active_response_output, hidden_active_output = tf.nn.dynamic_rnn(lstm_active,
                                                                                                  tf.expand_dims(
                                                                                                      fc_active_response,
                                                                                                      1),
                                                                                                  initial_state=hidden_active_output,
                                                                                                  sequence_length=tf.ones(
                                                                                                      [
                                                                                                          tf.shape(
                                                                                                              state_id)[
                                                                                                              0]]))
                            fc_active_decision = res_fc_block(tf.squeeze(lstm_active_response_output, axis=[1]), 64)
                            active_response_logits = slim.fully_connected(inputs=res_fc_block(fc_active_decision, 64), num_outputs=15,
                                                                          activation_fn=None)

                        with tf.variable_scope('seq_length'):
                            fc_active_seq = res_fc_block(flattened, 256)
                            lstm_active_seq_output, _ = tf.nn.dynamic_rnn(lstm_active,
                                                                          tf.expand_dims(fc_active_seq, 1),
                                                                          initial_state=hidden_active_output,
                                                                          sequence_length=tf.ones(
                                                                              [tf.shape(state_id)[0]]))
                            fc_active_seq = res_fc_block(tf.squeeze(lstm_active_seq_output, axis=[1]), 64)
                            active_seq_logits = slim.fully_connected(inputs=res_fc_block(fc_active_seq, 64), num_outputs=12,
                                                                     activation_fn=None)

                    with tf.variable_scope('branch_minor'):
                        fc_minor = res_fc_block(flattened, 256)
                        minor_type_embedding = slim.fully_connected(inputs=res_fc_block(tf.one_hot(minor_type_id, 2), 256), num_outputs=256,
                                                                    activation_fn=tf.nn.sigmoid)
                        fc_minor = fc_minor * minor_type_embedding

                        fc_minor = res_fc_block(fc_minor, 64)
                        minor_response_logits = slim.fully_connected(inputs=res_fc_block(fc_minor, 64), num_outputs=15,
                                                                     activation_fn=None)
            gathered_output = [passive_decision_logits, passive_bomb_logits, passive_response_logits,
                   active_decision_logits, active_response_logits, active_seq_logits, minor_response_logits]
            if idx == 1 or idx == 3:
                for k in range(len(gathered_output)):
                    gathered_output[k] = tf.stop_gradient(gathered_output[k])
            gathered_outputs.append(gathered_output)

        # 7: B * ?
        outputs = []
        for i in range(7):
            scatter_shape = tf.cast(tf.stack([batch_size, gathered_outputs[0][i].shape[1]]), dtype=tf.int64)
            # scatter_shape = tf.Print(scatter_shape, [tf.shape(scatter_shape)])
            outputs.append(tf.add_n([tf.scatter_nd(indices[k], gathered_outputs[k][i], scatter_shape) for k in range(3)]))

        return outputs

    def get_value(self, role_id, state):
        with tf.variable_scope('value_network'):
            # not adding regular loss for fc since we need big scalar output [-1, 1]
            with tf.variable_scope('value_fc'):
                x = state
                feats = [1024, 512, 512, 256, 256]
                for f in feats:
                    for _ in range(3):
                        x = res_fc_block(x, f)
                flattened = x
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

        entropy_beta = tf.get_variable('entropy_beta', shape=[], initializer=tf.constant_initializer(0.001),
                                       trainable=False)

        # regularization loss
        ctx = get_current_tower_context()
        if ctx.has_own_variables:  # be careful of the first tower (name='')
            l2_loss = ctx.get_collection_in_tower(tf.GraphKeys.REGULARIZATION_LOSSES)
        else:
            l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(l2_loss) > 0:
            logger.info("regularize_cost_from_collection() found {} regularizers "
                        "in REGULARIZATION_LOSSES collection.".format(len(l2_loss)))

        # 3 * 7
        l2_losses = []
        for role in range(1, 4):
            scope = 'policy_network_%d' % role
            l2_loss_role = [l for l in l2_loss if l.op.name.startswith(scope)]
            l2_main_loss = [l for l in l2_loss_role if 'branch_main' in l.name]
            l2_passive_fc_loss = [l for l in l2_loss_role if
                                  'branch_passive' in l.name and 'decision' not in l.name and 'bomb' not in l.name and 'response' not in l.name]
            l2_active_fc_loss = [l for l in l2_loss_role if
                                 'branch_active' in l.name and 'decision' not in l.name and 'response' not in l.name and 'seq_length' not in l.name]
            l2_active_lstm_weight = [l for l in ctx.get_collection_in_tower(tf.GraphKeys.TRAINABLE_VARIABLES) if l.op.name == scope + '/branch_active/decision/rnn/basic_lstm_cell/kernel']
            l2_active_lstm_loss = [POLICY_WEIGHT_DECAY * tf.nn.l2_loss(l2_active_lstm_weight[0])]
            assert len(l2_active_lstm_loss) > 0

            print('l2 loss', len(l2_loss_role))
            print('l2 main loss', len(l2_main_loss))
            print('l2 passive fc loss', len(l2_passive_fc_loss))
            print('l2 active fc loss', len(l2_active_fc_loss))

            name_scopes = ['branch_passive/decision', 'branch_passive/bomb', 'branch_passive/response',
                           'branch_active/decision', 'branch_active/response', 'branch_active/seq_length',
                           'branch_minor']

            # 7
            losses = []
            for i, name in enumerate(name_scopes):
                l2_branch_loss = l2_main_loss.copy()
                if 'passive' in name:
                    if 'bomb' in name:
                        l2_branch_loss += [l for l in l2_loss_role if name in l.name]
                    else:
                        l2_branch_loss += l2_passive_fc_loss + [l for l in l2_loss_role if name in l.name]
                else:
                    if 'minor' in name:
                        # do not include lstm regularization in minor loss
                        l2_branch_loss += l2_active_fc_loss + [l for l in l2_loss_role if name in l.name]
                    else:
                        l2_branch_loss += l2_active_fc_loss + [l for l in l2_loss_role if name in l.name] + l2_active_lstm_loss

                losses.append(tf.add_n(l2_branch_loss))
                # print('losses shape', losses[i].shape)
                print(name, 'l2 branch loss', len(l2_branch_loss))
            losses = tf.stack(losses, axis=0)
            if role == 1 or role == 3:
                losses = tf.stop_gradient(losses)
            l2_losses.append(losses)

        # 3 * 7
        l2_losses = tf.stack(l2_losses, axis=0)

        # B * 7
        l2_losses = tf.gather(l2_losses, role_id)

        # B
        l2_losses = tf.gather_nd(l2_losses, idx)

        print(l2_losses.shape)
        # print(policy_loss_b.shape)
        # print(entropy_loss_b.shape)
        # print(value_loss_b.shape)
        # print(advantage_b.shape)
        costs = []
        for i in range(1, 4):
            mask = tf.equal(role_id, i)
            # print(mask.shape)
            l2_loss = tf.reduce_mean(tf.boolean_mask(l2_losses, mask), name='l2_loss_%d' % i)
            pred_reward = tf.reduce_mean(tf.boolean_mask(value, mask), name='predict_reward_%d' % i)
            true_reward = tf.reduce_mean(tf.boolean_mask(discounted_return, mask), name='true_reward_%d' % i)
            advantage = tf.sqrt(tf.reduce_mean(tf.square(tf.boolean_mask(advantage_b, mask))), name='rms_advantage_%d' % i)

            policy_loss = tf.reduce_sum(tf.boolean_mask(policy_loss_b, mask, name='policy_loss_%d' % i))
            entropy_loss = tf.reduce_sum(tf.boolean_mask(entropy_loss_b, mask, name='entropy_loss_%d' % i))
            value_loss = tf.reduce_sum(tf.boolean_mask(value_loss_b, mask, name='value_loss_%d' % i))
            cost = tf.add_n([policy_loss, entropy_loss * entropy_beta, value_loss, l2_loss])
            cost = tf.truediv(cost, tf.reduce_sum(tf.cast(mask, tf.float32)), name='cost_%d' % i)
            costs.append(cost)

            importance = tf.reduce_mean(tf.boolean_mask(importance_b, mask), name='importance_%d' % i)
            add_moving_summary(policy_loss, entropy_loss, value_loss, pred_reward, true_reward, advantage, cost, importance, decay=0)

        return tf.add_n(costs)

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.3))]
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
             'active_response_prob', 'active_seq_prob', 'minor_response_prob', 'mode_out'],
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
            # logger.info('async predictor callback')
            try:
                output = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(client.ident))
                return
            mode = output[-1]
            distrib = (output[:-1][mode] + 1e-6) * mask
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib / distrib.sum())
            client.memory[role_id - 1].append(TransitionExperience(
                prob_state, all_state, action, reward=0, minor_type=minor_type, first_st=first_st,
                last_cards_onehot=last_cards_onehot, mode=mode, prob=distrib[action]))
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
                if i != 1:
                    continue
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
            if role_id != 2:
                continue
            mem = client.memory[role_id - 1]

            mem.reverse()
            R = float(init_r)
            mem_valid = [m for m in mem if m.first_st]
            dr = []
            for idx, k in enumerate(mem_valid):
                R = k.reward + GAMMA * R
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
    dirname = os.path.join('train_log', 'a3c_small')
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
            HumanHyperParamSetter('learning_rate'),
            Evaluator(
                100, ['role_id', 'policy_state_in', 'last_cards_in', 'minor_type_in'],
                ['passive_decision_prob', 'passive_bomb_prob', 'passive_response_prob',
                 'active_decision_prob', 'active_response_prob', 'active_seq_prob', 'minor_response_prob'], get_player),
        ],
        # session_init=ModelLoader('policy_network_2', 'SL_policy_network', 'value_network', 'SL_value_network'),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )
    trainer = SimpleTrainer() if config.nr_tower == 1 else AsyncMultiGPUTrainer(train_tower)
    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    train()
