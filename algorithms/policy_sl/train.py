from tensorpack.tfutils.summary import add_moving_summary
from tensorpack import *
from tensorpack.tfutils.gradproc import MapGradient
from tensorpack.tfutils import (
    get_current_tower_context, optimizer)
from tensorpack.utils.gpu import get_nr_gpu
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.insert(0, ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))


from env import Env
from doudizhu.card import action_space, action_index
from doudizhu.utils import to_char
from algorithms.policy_sl.evaluator import Evaluator, policy_inputs, LSTM_STATE_DIM
import multiprocessing
import numpy as np
from algorithms.resnet_blocks import identity_block, upsample_block, downsample_block
import tensorflow as tf

INPUT_DIM = 60 * 3
LAST_INPUT_DIM = 60 * 2
WEIGHT_DECAY = 5 * 1e-4
SCOPE = 'SL_policy_network'
# residual layers of each conv tower, as [num_channel, kernel_size, type]
CONV_LAYERS = [[128, 3, 'identity'],
               [128, 3, 'identity'],
               [128, 3, 'downsampling'],
               [128, 3, 'identity'],
               [128, 3, 'identity'],
               [256, 3, 'downsampling'],
               [256, 3, 'identity'],
               [256, 3, 'identity']]

# number of games per epoch roughly = STEPS_PER_EPOCH * BATCH_SIZE / 100
STEPS_PER_EPOCH = 1000
BATCH_SIZE = 1024

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

        # conv_idens = []
        # for c in conv_list:
        #     for i in range(5):
        #         c = identity_block(c, 32, 3)
        #     conv_idens.append(c)
        # conv = tf.concat(conv_idens, -1)

        for param in res_params:
            if param[-1] == 'identity':
                conv = identity_block(conv, param[0], param[1])
            elif param[-1] == 'downsampling':
                conv = downsample_block(conv, param[0], param[1])
            elif param[-1] == 'upsampling':
                conv = upsample_block(conv, param[0], param[1])
            else:
                raise Exception('unsupported layer type')
        # assert conv.shape[1] * conv.shape[2] * conv.shape[3] == 1024
        conv = tf.reshape(conv, [-1, conv.shape[1] * conv.shape[2] * conv.shape[3]])
        # conv = tf.squeeze(tf.reduce_mean(conv, axis=[2]), axis=[1])
    return conv


def get_player():
    return Env()


class DataFromGeneratorRNG(RNGDataFlow):
    """
    Wrap a generator to a DataFlow.
    """
    def __init__(self, gen, size=None):
        """
        Args:
            gen: iterable, or a callable that returns an iterable
            size: deprecated
        """
        if not callable(gen):
            self._gen = lambda: gen
        else:
            self._gen = gen
        if size is not None:
            logger.warn("DataFromGenerator(size=)", "It doesn't make much sense.", "2018-03-31")

    def get_data(self):
        # yield from
        for dp in self._gen(self.rng):
            yield dp


def data_generator(rng):
    """Moves of the rule-based players, as (state, last_cards, lstm_state, action, mode) samples.

    Every sample is fed a zero LSTM state, i.e. the LSTM is pretrained as at the start of a game; A3C
    then carries the state from move to move.
    """
    env = Env(rng.randint(1 << 31))
    lstm_state = np.zeros([LSTM_STATE_DIM])

    while True:
        env.reset()
        env.prepare()
        r = 0
        while r == 0:
            state, last_cards, mode = policy_inputs(env)
            intention, r, _ = env.step_auto()
            action = action_index(to_char(intention))
            if action is not None:
                yield state, last_cards, lstm_state, action, mode


def policy_network(state, last_cards, lstm_state, weight_decay):
    """One player's policy network. A3C builds its policy networks with this function too, so ModelLoader
    can initialise them from this one.

    Returns the logits over action_space when leading (active) and when responding (passive), and the
    new LSTM state.
    """
    lstm = rnn.BasicLSTMCell(LSTM_STATE_DIM // 2, state_is_tuple=False)
    with slim.arg_scope([slim.fully_connected, slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with tf.variable_scope('branch_main'):
            flattened_1 = conv_block(state[:, :60], 32, INPUT_DIM // 3, CONV_LAYERS, 'branch_main1')
            flattened_2 = conv_block(state[:, 60:120], 32, INPUT_DIM // 3, CONV_LAYERS, 'branch_main2')
            flattened_3 = conv_block(state[:, 120:], 32, INPUT_DIM // 3, CONV_LAYERS, 'branch_main3')
            flattened = tf.concat([flattened_1, flattened_2, flattened_3], axis=1)

        fc, new_lstm_state = lstm(flattened, lstm_state)

        active_fc = slim.fully_connected(fc, 1024)
        active_logits = slim.fully_connected(active_fc, len(action_space), activation_fn=None, scope='final_fc')
        with tf.variable_scope('branch_passive'):
            flattened_last = conv_block(last_cards, 32, LAST_INPUT_DIM, CONV_LAYERS, 'last_cards')
            passive_attention = slim.fully_connected(inputs=flattened_last, num_outputs=1024,
                                                     activation_fn=tf.nn.sigmoid)
            passive_fc = passive_attention * active_fc
        passive_logits = slim.fully_connected(passive_fc, len(action_space), activation_fn=None, reuse=True, scope='final_fc')
    return active_logits, passive_logits, new_lstm_state


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, INPUT_DIM], 'state_in'),
                tf.placeholder(tf.float32, [None, LAST_INPUT_DIM], 'last_cards_in'),
                tf.placeholder(tf.float32, [None, LSTM_STATE_DIM], 'lstm_state_in'),
                tf.placeholder(tf.int32, [None], 'action_in'),
                tf.placeholder(tf.int32, [None], 'mode_in')
                ]

    def build_graph(self, state, last_cards, lstm_state, action_target, mode):
        with tf.variable_scope(SCOPE):
            active_logits, passive_logits, _ = policy_network(state, last_cards, lstm_state, WEIGHT_DECAY)
        tf.nn.softmax(active_logits, name='active_prob')
        tf.nn.softmax(passive_logits, name='passive_prob')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return

        # mode 0: leading, use the active logits; mode 1: responding, use the passive ones
        logits = tf.where(tf.equal(mode, 0), active_logits, passive_logits)
        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action_target, logits=logits)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), action_target),
                                          tf.float32), name='accuracy')

        # as in A3C, the active logits are not regularised through branch_passive
        ctx = get_current_tower_context()
        if ctx.has_own_variables:  # be careful of the first tower (name='')
            l2_loss = ctx.get_collection_in_tower(tf.GraphKeys.REGULARIZATION_LOSSES)
        else:
            l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_active_loss = tf.add_n([l for l in l2_loss if 'branch_passive' not in l.name])
        l2_passive_loss = tf.add_n(l2_loss)
        l2_loss = tf.gather(tf.stack([l2_active_loss, l2_passive_loss]), mode)

        xent_loss = tf.reduce_mean(xent_loss, name='xent_loss')
        loss = tf.add(xent_loss, tf.reduce_mean(l2_loss), name='loss')
        add_moving_summary(loss, xent_loss, accuracy, decay=0.1)
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.3))]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


def train():
    dirname = os.path.join('train_log', 'train-SL-1.4')
    logger.set_logger_dir(dirname)

    # assign GPUs for training & inference
    nr_gpu = get_nr_gpu()
    if nr_gpu > 0:
        train_tower = list(range(nr_gpu)) or [0]
        logger.info("[Batch-SL] Train on gpu {}".format(
            ','.join(map(str, train_tower))))
    else:
        logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
        train_tower = [0]

    dataflow = DataFromGeneratorRNG(data_generator)
    if os.name == 'nt':
        dataflow = PrefetchData(dataflow, nr_proc=multiprocessing.cpu_count() // 2, nr_prefetch=multiprocessing.cpu_count() // 2)
    else:
        dataflow = PrefetchDataZMQ(dataflow, nr_proc=multiprocessing.cpu_count() // 2)
    dataflow = BatchData(dataflow, BATCH_SIZE)
    config = TrainConfig(
        model=Model(),
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            EstimatedTimeLeft(),
            PeriodicTrigger(Evaluator(
                100, ['state_in', 'last_cards_in', 'lstm_state_in'], ['active_prob', 'passive_prob'], get_player),
                every_k_epochs=1),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=100,
    )
    trainer = AsyncMultiGPUTrainer(train_tower) if nr_gpu > 1 else SimpleTrainer()
    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    train()
