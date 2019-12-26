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

from card import action_space
from TensorPack.MA_Hierarchical_Q.env import Env
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import get_current_tower_context, optimizer

from TensorPack.A3C.simulator import SimulatorProcess, SimulatorMaster, TransitionExperience, ROLE_IDS_TO_TRAIN
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
POLICY_INPUT_DIM = 60 + 120

POLICY_LAST_INPUT_DIM = 60 * 2
POLICY_WEIGHT_DECAY = 1e-3
VALUE_INPUT_DIM = 60 * 3
LORD_ID = 2
SIMULATOR_PROC = 20

# number of games per epoch roughly = STEPS_PER_EPOCH * BATCH_SIZE / 100
STEPS_PER_EPOCH = 2500
BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 32
PREDICTOR_THREAD_PER_GPU = 4
PREDICTOR_THREAD = None


def get_player():
    return CEnv()


class Model(ModelDesc):
    def get_policy(self, role_id, state, last_cards, lstm_state):
        # policy network, different for three agents
        batch_size = tf.shape(role_id)[0]
        gathered_outputs = []
        indices = []
        # train landlord only
        for idx in range(1, 4):
            with tf.variable_scope('policy_network_%d' % idx):
                lstm = rnn.BasicLSTMCell(1024, state_is_tuple=False)
                id_idx = tf.where(tf.equal(role_id, idx))
                indices.append(id_idx)
                state_id = tf.gather_nd(state, id_idx)
                last_cards_id = tf.gather_nd(last_cards, id_idx)
                lstm_state_id = tf.gather_nd(lstm_state, id_idx)
                with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(POLICY_WEIGHT_DECAY)):
                    with tf.variable_scope('branch_main'):
                        flattened_1 = policy_conv_block(state_id[:, :60], 32, POLICY_INPUT_DIM // 3,
                                                        [[128, 3, 'identity'],
                                                         [128, 3, 'identity'],
                                                         [128, 3, 'downsampling'],
                                                         [128, 3, 'identity'],
                                                         [128, 3, 'identity'],
                                                         [256, 3, 'downsampling'],
                                                         [256, 3, 'identity'],
                                                         [256, 3, 'identity']
                                                         ], 'branch_main1')
                        flattened_2 = policy_conv_block(state_id[:, 60:120], 32, POLICY_INPUT_DIM // 3,
                                                        [[128, 3, 'identity'],
                                                         [128, 3, 'identity'],
                                                         [128, 3, 'downsampling'],
                                                         [128, 3, 'identity'],
                                                         [128, 3, 'identity'],
                                                         [256, 3, 'downsampling'],
                                                         [256, 3, 'identity'],
                                                         [256, 3, 'identity']
                                                         ], 'branch_main2')
                        flattened_3 = policy_conv_block(state_id[:, 120:], 32, POLICY_INPUT_DIM // 3,
                                                        [[128, 3, 'identity'],
                                                         [128, 3, 'identity'],
                                                         [128, 3, 'downsampling'],
                                                         [128, 3, 'identity'],
                                                         [128, 3, 'identity'],
                                                         [256, 3, 'downsampling'],
                                                         [256, 3, 'identity'],
                                                         [256, 3, 'identity']
                                                         ], 'branch_main3')

                        flattened = tf.concat([flattened_1, flattened_2, flattened_3], axis=1)

                    fc, new_lstm_state = lstm(flattened, lstm_state_id)

                    active_fc = slim.fully_connected(fc, 1024)
                    active_logits = slim.fully_connected(active_fc, len(action_space), activation_fn=None, scope='final_fc')
                    with tf.variable_scope('branch_passive'):
                        flattened_last = policy_conv_block(last_cards_id, 32, POLICY_LAST_INPUT_DIM,
                                                           [[128, 3, 'identity'],
                                                            [128, 3, 'identity'],
                                                            [128, 3, 'downsampling'],
                                                            [128, 3, 'identity'],
                                                            [128, 3, 'identity'],
                                                            [256, 3, 'downsampling'],
                                                            [256, 3, 'identity'],
                                                            [256, 3, 'identity']
                                                            ], 'last_cards')

                        passive_attention = slim.fully_connected(inputs=flattened_last, num_outputs=1024,
                                                                      activation_fn=tf.nn.sigmoid)
                        passive_fc = passive_attention * active_fc
                    passive_logits = slim.fully_connected(passive_fc, len(action_space), activation_fn=None, reuse=True, scope='final_fc')

            gathered_output = [active_logits, passive_logits, new_lstm_state]
            if idx not in ROLE_IDS_TO_TRAIN:
                for k in range(len(gathered_output)):
                    gathered_output[k] = tf.stop_gradient(gathered_output[k])
            gathered_outputs.append(gathered_output)

        # 3: B * ?
        outputs = []
        for i in range(3):
            scatter_shape = tf.cast(tf.stack([batch_size, gathered_outputs[0][i].shape[1]]), dtype=tf.int64)
            # scatter_shape = tf.Print(scatter_shape, [tf.shape(scatter_shape)])
            outputs.append(tf.add_n([tf.scatter_nd(indices[k], gathered_outputs[k][i], scatter_shape) for k in range(3)]))

        return outputs

    def get_value(self, role_id, state):
        with tf.variable_scope('value_network'):
            # not adding regular loss for fc since we need big scalar output [-1, 1]
            with tf.variable_scope('value_conv'):
                flattened_1 = value_conv_block(state[:, :60], 32, VALUE_INPUT_DIM // 3, [[128, 3, 'identity'],
                                                                  [128, 3, 'identity'],
                                                                  [128, 3, 'downsampling'],
                                                                  [128, 3, 'identity'],
                                                                  [128, 3, 'identity'],
                                                                  [256, 3, 'downsampling'],
                                                                  [256, 3, 'identity'],
                                                                  [256, 3, 'identity']
                                                                  ], 'value_conv1')
                flattened_2 = value_conv_block(state[:, 60:120], 32, VALUE_INPUT_DIM // 3, [[128, 3, 'identity'],
                                                                  [128, 3, 'identity'],
                                                                  [128, 3, 'downsampling'],
                                                                  [128, 3, 'identity'],
                                                                  [128, 3, 'identity'],
                                                                  [256, 3, 'downsampling'],
                                                                  [256, 3, 'identity'],
                                                                  [256, 3, 'identity']
                                                                  ], 'value_conv2')
                flattened_3 = value_conv_block(state[:, 120:], 32, VALUE_INPUT_DIM // 3, [[128, 3, 'identity'],
                                                                  [128, 3, 'identity'],
                                                                  [128, 3, 'downsampling'],
                                                                  [128, 3, 'identity'],
                                                                  [128, 3, 'identity'],
                                                                  [256, 3, 'downsampling'],
                                                                  [256, 3, 'identity'],
                                                                  [256, 3, 'identity']
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
            tf.placeholder(tf.int32, [None], 'action_in'),
            tf.placeholder(tf.int32, [None], 'mode_in'),
            tf.placeholder(tf.float32, [None], 'history_action_prob_in'),
            tf.placeholder(tf.float32, [None], 'discounted_return_in'),
            tf.placeholder(tf.float32, [None, 1024 * 2], 'lstm_state_in')
        ]

    def build_graph(self, role_id, prob_state, value_state, last_cards, action_target, mode, history_action_prob, discounted_return, lstm_state):

        active_logits, passive_logits, new_lstm_state = self.get_policy(role_id, prob_state, last_cards, lstm_state)
        new_lstm_state = tf.identity(new_lstm_state, name='new_lstm_state')
        active_prob = tf.nn.softmax(active_logits, name='active_prob')
        passive_prob = tf.nn.softmax(passive_logits, name='passive_prob')
        mode_out = tf.identity(mode, name='mode_out')
        value = self.get_value(role_id, value_state)
        # this is the value for each agent, not the global value
        value = tf.identity(value, name='pred_value')
        is_training = get_current_tower_context().is_training

        if not is_training:
            return

        action_target_onehot = tf.one_hot(action_target, len(action_space))

        # active mode
        active_logpa = tf.reduce_sum(action_target_onehot * tf.log(
            tf.clip_by_value(active_prob, 1e-7, 1 - 1e-7)), 1)

        # passive mode
        passive_logpa = tf.reduce_sum(action_target_onehot * tf.log(
            tf.clip_by_value(passive_prob, 1e-7, 1 - 1e-7)), 1)

        # B * 2
        logpa = tf.stack([active_logpa, passive_logpa], axis=1)
        idx = tf.stack([tf.range(tf.shape(prob_state)[0]), mode], axis=1)

        # B
        logpa = tf.gather_nd(logpa, idx)

        # importance sampling
        active_pa = tf.reduce_sum(action_target_onehot * tf.clip_by_value(active_prob, 1e-7, 1 - 1e-7), 1)
        passive_pa = tf.reduce_sum(action_target_onehot * tf.clip_by_value(passive_prob, 1e-7, 1 - 1e-7), 1)

        # B * 2
        pa = tf.stack([active_pa, passive_pa], axis=1)
        idx = tf.stack([tf.range(tf.shape(prob_state)[0]), mode], axis=1)

        # B
        pa = tf.gather_nd(pa, idx)

        # using PPO
        ppo_epsilon = tf.get_variable('ppo_epsilon', shape=[], initializer=tf.constant_initializer(0.2),
                                       trainable=False)
        importance_b = pa / (history_action_prob + 1e-8)

        # advantage
        advantage_b = tf.subtract(discounted_return, tf.stop_gradient(value), name='advantage')

        policy_loss_b = -tf.minimum(importance_b * advantage_b, tf.clip_by_value(importance_b, 1 - ppo_epsilon, 1 + ppo_epsilon) * advantage_b)
        entropy_loss_b = pa * logpa
        value_loss_b = tf.square(value - discounted_return)

        entropy_beta = tf.get_variable('entropy_beta', shape=[], initializer=tf.constant_initializer(0.005),
                                       trainable=False)

        value_weight = tf.get_variable('value_weight', shape=[], initializer=tf.constant_initializer(0.2), trainable=False)

        # regularization loss
        ctx = get_current_tower_context()
        if ctx.has_own_variables:  # be careful of the first tower (name='')
            l2_loss = ctx.get_collection_in_tower(tf.GraphKeys.REGULARIZATION_LOSSES)
        else:
            l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(l2_loss) > 0:
            logger.info("regularize_cost_from_collection() found {} regularizers "
                        "in REGULARIZATION_LOSSES collection.".format(len(l2_loss)))

        # 3 * 2
        l2_losses = []
        for role in range(1, 4):
            scope = 'policy_network_%d' % role
            l2_loss_role = [l for l in l2_loss if l.op.name.startswith(scope)]
            l2_active_loss = [l for l in l2_loss_role if 'branch_passive' not in l.name]
            l2_passive_loss = l2_loss_role
            print('l2 active loss: {}'.format(len(l2_active_loss)))
            print('l2 passive loss: {}'.format(len(l2_passive_loss)))

            # 2
            losses = [tf.add_n(l2_active_loss), tf.add_n(l2_passive_loss)]
            losses = tf.stack(losses, axis=0)
            if role == 1 or role == 3:
                losses = tf.stop_gradient(losses)
            l2_losses.append(losses)

        # 3 * 2
        l2_losses = tf.stack(l2_losses, axis=0)

        # B * 2
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
            valid_batch = tf.reduce_sum(tf.cast(mask, tf.float32))
            # print(mask.shape)
            l2_loss = tf.truediv(tf.reduce_sum(tf.boolean_mask(l2_losses, mask)), valid_batch, name='l2_loss_%d' % i)
            pred_reward = tf.truediv(tf.reduce_sum(tf.boolean_mask(value, mask)), valid_batch, name='predict_reward_%d' % i)
            true_reward = tf.truediv(tf.reduce_sum(tf.boolean_mask(discounted_return, mask)), valid_batch, name='true_reward_%d' % i)
            advantage = tf.sqrt(tf.truediv(tf.reduce_sum(tf.square(tf.boolean_mask(advantage_b, mask))), valid_batch), name='rms_advantage_%d' % i)

            policy_loss = tf.truediv(tf.reduce_sum(tf.boolean_mask(policy_loss_b, mask)), valid_batch, name='policy_loss_%d' % i)
            entropy_loss = tf.truediv(tf.reduce_sum(tf.boolean_mask(entropy_loss_b, mask)), valid_batch, name='entropy_loss_%d' % i)
            value_loss = tf.truediv(tf.reduce_sum(tf.boolean_mask(value_loss_b, mask)), valid_batch, name='value_loss_%d' % i)
            cost = tf.add_n([policy_loss, entropy_loss * entropy_beta, value_weight * value_loss, l2_loss], name='cost_%d' % i)
            # cost = tf.truediv(cost, tf.reduce_sum(tf.cast(mask, tf.float32)), name='cost_%d' % i)
            costs.append(cost)

            importance = tf.truediv(tf.reduce_sum(tf.boolean_mask(importance_b, mask)), valid_batch, name='importance_%d' % i)
            add_moving_summary(policy_loss, entropy_loss, value_loss, l2_loss, pred_reward, true_reward, advantage, cost, importance, decay=0)

        return tf.add_n(costs)

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5))]
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
            ['role_id', 'policy_state_in', 'value_state_in', 'last_cards_in', 'mode_in', 'lstm_state_in'],
            ['active_prob', 'passive_prob', 'mode_out', 'new_lstm_state'],
            self._gpus[k % nr_gpu])
            for k in range(PREDICTOR_THREAD)]
        self.async_predictor = MultiThreadAsyncPredictor(
            predictors, batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()

    def _on_state(self, role_id, prob_state, all_state, last_cards_onehot, mask, mode, lstm_state, client):
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
            new_lstm_state = output[-1]
            mode = output[-2]
            distrib = (output[:-2][mode] + 1e-7) * mask
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib / distrib.sum())
            client.memory[role_id - 1].append(TransitionExperience(
                prob_state, all_state, action, reward=0, lstm_state=lstm_state,
                last_cards_onehot=last_cards_onehot, mode=mode, prob=distrib[action]))
            self.send_queue.put([client.ident, dumps((action, new_lstm_state))])
        self.async_predictor.put_task([role_id, prob_state, all_state, last_cards_onehot, mode, lstm_state], cb)

    def _process_msg(self, client, role_id, prob_state, all_state, last_cards_onehot, mask, mode, lstm_state, reward, isOver):
        """
        Process a message sent from some client.
        """
        # in the first message, only state is valid,
        # reward&isOver should be discarde
        if isOver:
            # should clear client's memory and put to queue
            assert reward != 0
            for i in range(3):
                if i != 1:
                    continue
                    # notice that C++ returns the reward for farmer, transform to the reward in each agent's perspective
                client.memory[i][-1].reward = reward if i != 1 else -reward
            self._parse_memory(0, client)
        # feed state and return action
        self._on_state(role_id, prob_state, all_state, last_cards_onehot, mask, mode, lstm_state, client)

    def _parse_memory(self, init_r, client):
        # for each agent's memory
        for role_id in range(1, 4):
            if role_id not in ROLE_IDS_TO_TRAIN:
                continue
            mem = client.memory[role_id - 1]

            mem.reverse()
            R = float(init_r)
            for idx, k in enumerate(mem):
                R = k.reward + GAMMA * R
                self.queue.put([role_id, k.prob_state, k.all_state, k.last_cards_onehot, k.action, k.mode, k.prob, R, k.lstm_state])

            client.memory[role_id - 1] = []


def train():
    dirname = os.path.join('train_log', 'A3C-LSTM')
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
    if os.name == 'nt':
        namec2s = 'tcp://127.0.0.1:8000'
        names2c = 'tcp://127.0.0.1:9000'
    else:
        prefix = '@' if sys.platform.startswith('linux') else ''
        namec2s = 'ipc://{}sim-c2s-{}'.format(prefix, name_base)
        names2c = 'ipc://{}sim-s2c-{}'.format(prefix, name_base)

    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, predict_tower)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)
    config = AutoResumeTrainConfig(
        always_resume=True,
        # starting_epoch=0,
        model=Model(),
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            MaxSaver('true_reward_2'),
            HumanHyperParamSetter('learning_rate'),
            # ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            # ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            master,
            StartProcOrThread(master),
            Evaluator(
                100, ['role_id', 'policy_state_in', 'last_cards_in', 'lstm_state_in'],
                ['active_prob', 'passive_prob', 'new_lstm_state'], get_player),
            # SendStat(
            #     'export http_proxy=socks5://127.0.0.1:1080 https_proxy=socks5://127.0.0.1:1080 && /home/neil/anaconda3/bin/curl --header "Access-Token: o.CUdAMXqiVz9qXTxLYIXc0XkcAfZMpNGM" -d type=note -d title="doudizhu" '
            #     '-d body="lord win rate: {lord_win_rate}\n policy loss: {policy_loss_2}\n value loss: {value_loss_2}\n entropy loss: {entropy_loss_2}\n'
            #     'true reward: {true_reward_2}\n predict reward: {predict_reward_2}\n advantage: {rms_advantage_2}\n" '
            #     '--request POST https://api.pushbullet.com/v2/pushes',
            #     ['lord_win_rate', 'policy_loss_2', 'value_loss_2', 'entropy_loss_2',
            #      'true_reward_2', 'predict_reward_2', 'rms_advantage_2']
            #     ),
        ],
        # session_init=SaverRestore('./train_log/a3c_action_1d/max-true_reward_2'),
        # session_init=ModelLoader('policy_network_2', 'SL_policy_network', 'value_network', 'SL_value_network'),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )
    trainer = SimpleTrainer() if config.nr_tower == 1 else AsyncMultiGPUTrainer(train_tower)
    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    train()

