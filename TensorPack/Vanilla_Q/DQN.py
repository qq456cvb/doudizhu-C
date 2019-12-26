#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py

import os
import argparse
import cv2
import tensorflow as tf


from tensorpack import *
import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))
from TensorPack.Vanilla_Q.DQNModel import Model as DQNModel
from env import Env as CEnv
from card import action_space
import tensorflow.contrib.slim as slim
from TensorPack.Vanilla_Q.expreplay import ExpReplay
from TensorPack.ResNetBlock import identity_block, upsample_block, downsample_block
from TensorPack.Vanilla_Q.evaluator import Evaluator
from TensorPack.PolicySL.Policy_SL_v1_4 import conv_block


def res_fc_block(inputs, units, stack=3):
    residual = inputs
    for i in range(stack):
        residual = FullyConnected('fc%d' % i, residual, units, activation=tf.nn.relu)
    x = inputs
    # x = FullyConnected('fc', x, units, activation=tf.nn.relu)
    if inputs.shape[1].value != units:
        x = FullyConnected('fc', x, units, activation=tf.nn.relu)
    return tf.contrib.layers.layer_norm(residual + x, scale=False)
    # return residual + x


BATCH_SIZE = 4
STATE_SHAPE = (60 + 120 + 256 * 2,)
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 1e3
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = MEMORY_SIZE // 20
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ  # each epoch is 100k played frames
EVAL_EPISODE = 100

NUM_ACTIONS = None
METHOD = None


def get_player():
    return CEnv()


class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(STATE_SHAPE, METHOD, NUM_ACTIONS, GAMMA)

    def get_state_embedding(self, state):
        with tf.variable_scope('state_embedding'):
            shape = state.shape.as_list()
            net = tf.reshape(state, [-1, shape[-1]])
            net = tf.concat([conv_block(net[:, :180], 32, 180,
                                        [[128, 3, 'identity'],
                                         [128, 3, 'identity'],
                                         [128, 3, 'downsampling'],
                                         [128, 3, 'identity'],
                                         [128, 3, 'identity'],
                                         [256, 3, 'downsampling'],
                                         [256, 3, 'identity'],
                                         [256, 3, 'identity']
                                         ], 'handcards'), net[:, 180:]], -1)
            units = [512, 256, 128]
            for i, unit in enumerate(units):
                with tf.variable_scope('block%i' % i):
                    net = res_fc_block(net, unit)
            return net

    def get_action_embedding(self, action):
        with tf.variable_scope('action_embedding'):
            shape = tf.shape(action)
            net = tf.reshape(action, [-1, shape[-1]])
            units = [512, 256, 128]
            for i, unit in enumerate(units):
                with tf.variable_scope('block%i' % i):
                    net = res_fc_block(net, unit)
            return net

    def _get_DQN_prediction(self, state):
        shape = state.shape.as_list()
        net = tf.reshape(state, [-1, shape[-1]])
        net = tf.concat([conv_block(net[:, :180], 32, 180,
                          [[128, 3, 'identity'],
                           [128, 3, 'identity'],
                           [128, 3, 'downsampling'],
                           [128, 3, 'identity'],
                           [128, 3, 'identity'],
                           [256, 3, 'downsampling'],
                           [256, 3, 'identity'],
                           [256, 3, 'identity']
                           ], 'handcards'), net[:, 180:]], -1)
        units = [512, 512, 256, 256, 128, 128]
        for i, unit in enumerate(units):
            with tf.variable_scope('block%i' % i):
                net = res_fc_block(net, unit)
        l = net

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, len(action_space))
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, len(action_space))
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')


def get_config():
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(),
        state_shape=STATE_SHAPE,
        num_actions=NUM_ACTIONS,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.,
        update_frequency=UPDATE_FREQ
    )

    # ds = FakeData([(2, 2, *STATE_SHAPE), [2], [2], [2], [2]], dtype=['float32', 'int64', 'float32', 'bool', 'bool'])
    # ds = PrefetchData(ds, nr_prefetch=6, nr_proc=2)
    return AutoResumeTrainConfig(
        # always_resume=False,
        data=QueueInput(expreplay),
        model=Model(),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                every_k_steps=STEPS_PER_EPOCH // 10),    # update target network every 10k steps
            expreplay,
            # ScheduledHyperParamSetter('learning_rate',
            #                           [(60, 5e-5), (100, 2e-5)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                [(0, 1), (30, 0.5), (60, 0.1), (320, 0.01)],   # 1->0.1 in the first million steps
                interp='linear'),
            Evaluator(
                EVAL_EPISODE, ['state'], ['Qvalue'], get_player),
            HumanHyperParamSetter('learning_rate'),
        ],
        # starting_epoch=30,
        # session_init=SaverRestore('train_log/DQN-54-AUG-STATE/model-75000'),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling'], default='Double')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    METHOD = args.algo
    # set num_actions
    NUM_ACTIONS = len(action_space)

    nr_gpu = get_nr_gpu()
    train_tower = list(range(nr_gpu))
    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state', 'comb_mask'],
            output_names=['Qvalue']))
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'Vanilla-DQN'))
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SimpleTrainer() if nr_gpu == 1 else AsyncMultiGPUTrainer(train_tower)
        launch_train_with_config(config, trainer)
