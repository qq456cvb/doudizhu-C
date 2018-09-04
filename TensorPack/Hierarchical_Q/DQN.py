#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import argparse
import cv2
import tensorflow as tf


from tensorpack import *
import sys
import os
if os.name == 'nt':
    sys.path.insert(0, '../../build/Release')
else:
    sys.path.insert(0, '../../build.linux')
sys.path.insert(0, '../..')
from TensorPack.Hierarchical_Q.DQNModel import Model as DQNModel
from env import Env as CEnv
from card import action_space
import tensorflow.contrib.slim as slim
from TensorPack.Hierarchical_Q.expreplay import ExpReplay
from TensorPack.ResNetBlock import identity_block, upsample_block, downsample_block
from TensorPack.Hierarchical_Q.evaluator import Evaluator


def conv_block(input, conv_dim, input_dim, res_params, scope):
    with tf.variable_scope(scope):
        input_conv = tf.reshape(input, [-1, 1, input_dim, 1])
        single_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=conv_dim,
                                  kernel_size=[1, 1], stride=[1, 4], padding='SAME')

        pair_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=conv_dim,
                                kernel_size=[1, 2], stride=[1, 4], padding='SAME')

        triple_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=conv_dim,
                                  kernel_size=[1, 3], stride=[1, 4], padding='SAME')

        quadric_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=conv_dim,
                                   kernel_size=[1, 4], stride=[1, 4], padding='SAME')

        conv_list = [single_conv, pair_conv, triple_conv, quadric_conv]
        conv = tf.concat(conv_list, -1)

        for param in res_params:
            if param[-1] == 'identity':
                conv = identity_block(conv, param[0], param[1])
            elif param[-1] == 'downsampling':
                conv = downsample_block(conv, param[0], param[1])
            else:
                raise Exception('unsupported layer type')
        assert conv.shape[1] * conv.shape[2] * conv.shape[3] == 1024
        conv = tf.reshape(conv, [-1, conv.shape[1] * conv.shape[2] * conv.shape[3]])
        # conv = tf.squeeze(tf.reduce_mean(conv, axis=[2]), axis=[1])
    return conv


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

BATCH_SIZE = 8
MAX_NUM_COMBS = 100
MAX_NUM_GROUPS = 21
ATTEN_STATE_SHAPE = 60
HIDDEN_STATE_DIM = 256 + 256 + 120
STATE_SHAPE = (MAX_NUM_COMBS, MAX_NUM_GROUPS, HIDDEN_STATE_DIM)
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 1e4
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = MEMORY_SIZE // 20
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ  # each epoch is 100k played frames
EVAL_EPISODE = 100

NUM_ACTIONS = None
ROM_FILE = None
METHOD = None


def get_player():
    return CEnv()


class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(STATE_SHAPE, METHOD, NUM_ACTIONS, GAMMA)

    # input :B * COMB * N * D
    # output : B * COMB * D
    # assume N is padded with big negative numbers
    def _get_global_feature(self, joint_state):
        shape = joint_state.shape.as_list()
        net = tf.reshape(joint_state, [-1, shape[-1]])
        units = [256, 512, 1024]
        for i, unit in enumerate(units):
            with tf.variable_scope('block%i' % i):
                net = res_fc_block(net, unit)
        net = tf.reshape(net, [-1, shape[1], shape[2], units[-1]])
        return tf.reduce_max(net, [2])

    def _get_DQN_prediction_comb(self, state):
        shape = state.shape.as_list()
        net = tf.reshape(state, [-1, shape[-1]])
        units = [512, 256, 128]
        for i, unit in enumerate(units):
            with tf.variable_scope('block%i' % i):
                net = res_fc_block(net, unit)
        l = net

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, 1)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, 1)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.reshape(Q, [-1, shape[1], 1])

    def _get_DQN_prediction_fine(self, state):
        shape = state.shape.as_list()
        net = tf.reshape(state, [-1, shape[-1]])
        units = [512, 256, 128]
        for i, unit in enumerate(units):
            with tf.variable_scope('block%i' % i):
                net = res_fc_block(net, unit)
        l = net

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, 1)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, 1)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.reshape(Q, [-1, shape[1], 1])


def get_config():
    expreplay = ExpReplay(
        predictor_io_names=(['state', 'comb_mask', 'fine_mask'], ['Qvalue']),
        player=get_player(),
        state_shape=STATE_SHAPE,
        num_actions=[MAX_NUM_COMBS, MAX_NUM_GROUPS],
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
                EVAL_EPISODE, ['state', 'comb_mask', 'fine_mask'], ['Qvalue'], [MAX_NUM_COMBS, MAX_NUM_GROUPS], get_player),
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
    NUM_ACTIONS = max(MAX_NUM_GROUPS, MAX_NUM_COMBS)

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
            os.path.join('train_log', 'DQN-9-3-LASTCARDS'))
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SimpleTrainer() if nr_gpu == 1 else AsyncMultiGPUTrainer(train_tower)
        launch_train_with_config(config, trainer)
