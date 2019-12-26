from tensorpack.tfutils.summary import add_moving_summary
from tensorpack import *
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.tfutils import (
    get_current_tower_context, optimizer)
from tensorpack.utils.gpu import get_nr_gpu
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))


from env import Env
from card import Card, Category
from TensorPack.PolicySL.evaluator import Evaluator
from utils import get_seq_length, pick_minor_targets, to_char, discard_onehot_from_s_60
from utils import pick_main_cards
import multiprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim
from TensorPack.ResNetBlock import identity_block, upsample_block, downsample_block


INPUT_DIM = 60 * 3
LAST_INPUT_DIM = 60
WEIGHT_DECAY = 5 * 1e-4
SCOPE = 'SL_value_network'

# number of games per epoch roughly = STEPS_PER_EPOCH * BATCH_SIZE / 100
STEPS_PER_EPOCH = 1000
BATCH_SIZE = 2048
GAMMA = 0.99


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
    env = Env(rng.randint(1 << 31))
    # logger.info('called')

    while True:
        env.reset()
        env.prepare()
        r = 0
        buffer = []
        while r == 0:
            s = env.get_state_all_cards()
            buffer.append(s)
            intention, r, category_idx = env.step_auto()

        # convert reward to indicator function
        r = 1 if r > 0 else -1
        # negative reward means lord wins
        # assume discount factor 1
        for s in reversed(buffer):
            yield s, r
            r = GAMMA * r

    # dump for testing
    # env.reset()
    # env.prepare()
    # r = 0
    # buffer = []
    # while r == 0:
    #     s = env.get_state_all_cards()
    #     buffer.append(s)
    #     intention, r, category_idx = env.step_auto()
    #
    # # convert reward to indicator function
    # r = 1 if r > 0 else -1
    # # negative reward means lord wins
    # # assume discount factor 1
    # while True:
    #     for s in buffer:
    #         yield s, r


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
        assert conv.shape[1] * conv.shape[2] * conv.shape[3] == 1024
        conv = tf.reshape(conv, [-1, conv.shape[1] * conv.shape[2] * conv.shape[3]])
        # conv = tf.squeeze(tf.reduce_mean(conv, axis=[2]), axis=[1])
    return conv


class Model(ModelDesc):
    def get_pred(self, state):
        with tf.variable_scope(SCOPE):
            # not adding regular loss for fc since we need big scalar output [-1, 1]
            with slim.arg_scope([slim.conv2d]):
                with tf.variable_scope('value_conv'):
                    flattened_1 = conv_block(state[:, :60], 32, INPUT_DIM // 3, [[16, 32, 5, 'identity'],
                                                                      [16, 32, 5, 'identity'],
                                                                      [32, 128, 5, 'downsampling'],
                                                                      [32, 128, 3, 'identity'],
                                                                      [32, 128, 3, 'identity'],
                                                                      [64, 256, 3, 'downsampling'],
                                                                      [64, 256, 3, 'identity'],
                                                                      [64, 256, 3, 'identity']
                                                                      ], 'value_conv1')
                    flattened_2 = conv_block(state[:, 60:120], 32, INPUT_DIM // 3, [[16, 32, 5, 'identity'],
                                                                      [16, 32, 5, 'identity'],
                                                                      [32, 128, 5, 'downsampling'],
                                                                      [32, 128, 3, 'identity'],
                                                                      [32, 128, 3, 'identity'],
                                                                      [64, 256, 3, 'downsampling'],
                                                                      [64, 256, 3, 'identity'],
                                                                      [64, 256, 3, 'identity']
                                                                      ], 'value_conv2')
                    flattened_3 = conv_block(state[:, 120:], 32, INPUT_DIM // 3, [[16, 32, 5, 'identity'],
                                                                      [16, 32, 5, 'identity'],
                                                                      [32, 128, 5, 'downsampling'],
                                                                      [32, 128, 3, 'identity'],
                                                                      [32, 128, 3, 'identity'],
                                                                      [64, 256, 3, 'downsampling'],
                                                                      [64, 256, 3, 'identity'],
                                                                      [64, 256, 3, 'identity']
                                                                      ], 'value_conv3')
                    flattened = tf.concat([flattened_1, flattened_2, flattened_3], axis=1)

                with tf.variable_scope('value_fc'):
                    value = slim.fully_connected(flattened, num_outputs=1, activation_fn=None)
            return value

    def inputs(self):
        return [tf.placeholder(tf.float32, [None, INPUT_DIM], 'state_in'),
                tf.placeholder(tf.float32, [None], 'value_in')
                ]

    def build_graph(self, state, val_target):
        values = tf.squeeze(self.get_pred(state), axis=1, name='value')
        # fake_values = tf.zeros_like(values)
        is_training = get_current_tower_context().is_training
        if not is_training:
            return

        with tf.variable_scope("value_loss"):
            value_loss = tf.reduce_mean(tf.squared_difference(val_target, values), name='value_loss')
        # l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=SCOPE)
        # NOTE: this collection doesn't always grow with towers.
        # It only grows with actual variable creation, but not get_variable call.
        # ctx = get_current_tower_context()
        # if ctx.has_own_variables:  # be careful of the first tower (name='')
        #     l2_loss = ctx.get_collection_in_tower(tf.GraphKeys.REGULARIZATION_LOSSES)
        # else:
        #     l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # if len(l2_loss) > 0:
        #     logger.info("regularize_cost_from_collection() found {} regularizers "
        #                 "in REGULARIZATION_LOSSES collection.".format(len(l2_loss)))

        loss = value_loss
        loss = tf.identity(loss, name='loss')

        add_moving_summary(loss, decay=0)
        add_moving_summary(value_loss, decay=0)
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        # gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5)),
        #              SummaryGradient()]
        gradprocs = [MapGradient(lambda grad: grad),
                     SummaryGradient()]
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
        dataflow = PrefetchData(dataflow, nr_proc=multiprocessing.cpu_count(), nr_prefetch=multiprocessing.cpu_count())
    else:
        dataflow = PrefetchDataZMQ(dataflow, nr_proc=multiprocessing.cpu_count())
    dataflow = BatchData(dataflow, BATCH_SIZE)
    config = TrainConfig(
        model=Model(),
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            EstimatedTimeLeft(),
            # ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            # ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            # HumanHyperParamSetter('learning_rate'),
            # HumanHyperParamSetter('entropy_beta')
            # PeriodicTrigger(Evaluator(
            #     100, ['state_in', 'last_cards_in', 'minor_type_in'],
            #     ['passive_decision_prob', 'passive_bomb_prob', 'passive_response_prob',
            #      'active_decision_prob', 'active_response_prob', 'active_seq_prob', 'minor_response_prob'], get_player),
            #     every_k_epochs=1),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )
    trainer = AsyncMultiGPUTrainer(train_tower) if nr_gpu > 1 else SimpleTrainer()
    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    train()
    # e = Env()
    # for i in range(10000):
    #     e.reset()
    #     e.prepare()
    #     r = 0
    #     while r == 0:
    #         intention, r, _ = e.step_auto()
    #         print(e.get_state_all_cards())
            # print(intention)