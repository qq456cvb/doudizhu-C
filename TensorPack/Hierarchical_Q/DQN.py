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


BATCH_SIZE = 64
STATE_SHAPE = 180
ATTEN_STATE_SHAPE = 60
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 1e6
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = MEMORY_SIZE // 20
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ * 10  # each epoch is 100k played frames
EVAL_EPISODE = 50

NUM_ACTIONS = None
ROM_FILE = None
METHOD = None


def get_player():
    return CEnv()


class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(STATE_SHAPE, ATTEN_STATE_SHAPE, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, state, atten_state, is_active):

        flattened_1 = conv_block(state[:, :60], 32, STATE_SHAPE // 3,
                                        [[128, 3, 'identity'],
                                         [128, 3, 'identity'],
                                         [128, 3, 'downsampling'],
                                         [128, 3, 'identity'],
                                         [128, 3, 'identity'],
                                         [256, 3, 'downsampling'],
                                         [256, 3, 'identity'],
                                         [256, 3, 'identity']
                                         ], 'branch_main1')

        flattened = flattened_1

        fc = FullyConnected('fctAll', flattened, 1024, activation=tf.nn.relu)
        passive_idx = tf.where(tf.logical_not(is_active))
        active_idx = tf.where(is_active)
        with tf.variable_scope('branch_passive'):
            flattened_last = conv_block(tf.gather_nd(atten_state, passive_idx), 32, ATTEN_STATE_SHAPE,
                                               [[128, 3, 'identity'],
                                                [128, 3, 'identity'],
                                                [128, 3, 'downsampling'],
                                                [128, 3, 'identity'],
                                                [128, 3, 'identity'],
                                                [256, 3, 'downsampling'],
                                                [256, 3, 'identity'],
                                                [256, 3, 'identity']
                                                ], 'last_cards')

            passive_attention = FullyConnected('fctPassive', flattened_last, 1024,
                                                     activation=tf.nn.sigmoid)
            passive_fc = passive_attention * tf.gather_nd(fc, passive_idx)
        active_fc = tf.gather_nd(fc, active_idx)
        scatter_shape = tf.cast(tf.stack([tf.shape(fc)[0], 1024]), tf.int64)
        l = tf.scatter_nd(active_idx, active_fc, scatter_shape) + tf.scatter_nd(passive_idx, passive_fc, scatter_shape)
        l = tf.reshape(l, [-1, 1024])

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, self.num_actions)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')


def get_config():
    expreplay = ExpReplay(
        predictor_io_names=(['state', 'atten_state', 'is_active'], ['Qvalue']),
        player=get_player(),
        state_shape=STATE_SHAPE,
        atten_state_shape=ATTEN_STATE_SHAPE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ
    )

    return TrainConfig(
        data=QueueInput(expreplay),
        model=Model(),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                every_k_steps=10000 // UPDATE_FREQ),    # update target network every 10k steps
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      [(60, 4e-4), (100, 2e-4)]),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                [(0, 1), (10, 0.1), (320, 0.01)],   # 1->0.1 in the first million steps
                interp='linear'),
            # PeriodicTrigger(Evaluator(
            #     EVAL_EPISODE, ['state'], ['Qvalue'], get_player),
            #     every_k_epochs=10),
            HumanHyperParamSetter('learning_rate'),
        ],
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

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['Qvalue']))
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'DQN'))
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
