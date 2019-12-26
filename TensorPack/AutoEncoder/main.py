
import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

from tensorpack.tfutils.summary import add_moving_summary
from tensorpack import *
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.tfutils import (
    get_current_tower_context, optimizer)
from tensorpack.utils.gpu import get_nr_gpu
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import sys
import os
from card import Card, Category, action_space
from utils import get_seq_length, pick_minor_targets, to_char, discard_onehot_from_s_60
from utils import pick_main_cards
import multiprocessing
from TensorPack.ResNetBlock import identity_block, upsample_block, downsample_block
import tensorpack.dataflow
import tensorflow as tf
import numpy as np

INPUT_DIM = 60
STEPS_PER_EPOCH = 100
BATCH_SIZE = 256


class MyDataFlow(RNGDataFlow):
    def get_data(self):
        action_space_onehot = [Card.char2onehot60(a) for a in action_space]
        while True:
            yield [action_space_onehot[self.rng.randint(0, len(action_space_onehot))]]


class Evaluator(Callback):
    def __init__(self, input_names, output_names):
        self.action_space_onehot = np.array([Card.char2onehot60(a) for a in action_space])
        self.input_names = input_names
        self.output_names = output_names

    def _setup_graph(self):
        self.predictor = self.trainer.get_predictor(
            self.input_names, self.output_names)

    def _before_train(self):
        encoding, loss = self.predictor([self.action_space_onehot])
        encoding = np.squeeze(encoding, [1, 2])
        print(encoding.shape)
        print('loss: {}'.format(loss))
        np.save('encoding.npy', encoding)
        print('saved')


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, INPUT_DIM], 'state_in')]

    def build_graph(self, onehot_cards):
        scope = 'AutoEncoder'
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                weights_regularizer=slim.l2_regularizer(1e-3)):
                input_conv = tf.reshape(onehot_cards, [-1, 1, INPUT_DIM, 1])
                single_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=32,
                                          kernel_size=[1, 1], stride=[1, 4], padding='SAME')

                pair_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=32,
                                        kernel_size=[1, 2], stride=[1, 4], padding='SAME')

                triple_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=32,
                                          kernel_size=[1, 3], stride=[1, 4], padding='SAME')

                quadric_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=32,
                                           kernel_size=[1, 4], stride=[1, 4], padding='SAME')

                conv = tf.concat([single_conv, pair_conv, triple_conv, quadric_conv], -1)

                encoding_params = [[128, 3, 'identity'],
                                   [128, 3, 'identity'],
                                   [128, 3, 'downsampling'],
                                   [128, 3, 'identity'],
                                   [128, 3, 'identity'],
                                   [256, 3, 'downsampling'],
                                   [256, 3, 'identity'],
                                   [256, 3, 'identity']
                                   ]
                for param in encoding_params:
                    if param[-1] == 'identity':
                        conv = identity_block(conv, param[0], param[1])
                    elif param[-1] == 'upsampling':
                        conv = upsample_block(conv, param[0], param[1])
                    elif param[-1] == 'downsampling':
                        conv = downsample_block(conv, param[0], param[1])
                    else:
                        raise Exception('unsupported layer type')
                conv = tf.reduce_mean(conv, [1, 2], True)
                encoding = tf.identity(conv, name='encoding')

                # is_training = get_current_tower_context().is_training
                # if not is_training:
                #     return

                decoding_params = [[256, 4, 'upsampling'],
                                   [256, 3, 'identity'],
                                   [256, 3, 'identity'],
                                   [256, 4, 'upsampling'],
                                   [128, 3, 'identity'],
                                   [128, 3, 'identity'],
                                   [128, 4, 'upsampling'],
                                   [128, 3, 'identity'],
                                   [1, 3, 'identity']
                                   ]
                for param in decoding_params:
                    if param[-1] == 'identity':
                        conv = identity_block(conv, param[0], param[1])
                    elif param[-1] == 'upsampling':
                        conv = upsample_block(conv, param[0], param[1])
                    elif param[-1] == 'downsampling':
                        conv = downsample_block(conv, param[0], param[1])
                    else:
                        raise Exception('unsupported layer type')
                print(conv.shape)
                decoded = tf.reshape(conv, [-1, conv.shape[1] * conv.shape[2] * conv.shape[3]])

        reconstuct_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.pad(onehot_cards, [[0, 0], [0, 4]]), logits=decoded)
        reconstuct_loss = tf.reduce_mean(tf.reduce_sum(reconstuct_loss, -1), name='reconstruct_loss')
        l2_loss = tf.truediv(regularize_cost_from_collection(), tf.cast(tf.shape(onehot_cards)[0], tf.float32), name='l2_loss')
        add_moving_summary(reconstuct_loss, decay=0)
        add_moving_summary(l2_loss, decay=0)
        loss = reconstuct_loss + l2_loss
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.3))]
        # SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


def train():
    dirname = os.path.join('train_log', 'auto_encoder')
    logger.set_logger_dir(dirname)

    # assign GPUs for training & inference
    nr_gpu = get_nr_gpu()
    if nr_gpu > 0:
        train_tower = list(range(nr_gpu)) or [0]
        logger.info("[Batch-SL] Train on gpu {}".format(
            ','.join(map(str, train_tower))))
    else:
        logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
        train_tower = [0], [0]

    dataflow = MyDataFlow()
    if os.name == 'nt':
        dataflow = PrefetchData(dataflow, nr_proc=multiprocessing.cpu_count() // 2,
                                nr_prefetch=multiprocessing.cpu_count() // 2)
    else:
        dataflow = PrefetchDataZMQ(dataflow, nr_proc=multiprocessing.cpu_count() // 2)
    dataflow = BatchData(dataflow, BATCH_SIZE)
    config = TrainConfig(
        model=Model(),
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            EstimatedTimeLeft(),
            Evaluator(['state_in'], ['AutoEncoder/encoding', 'reconstruct_loss']),
            # ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            # ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            # HumanHyperParamSetter('learning_rate'),
        ],
        session_init=SaverRestore('train_log/auto_encoder/model-8000'),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=100,
    )
    trainer = AsyncMultiGPUTrainer(train_tower) if nr_gpu > 1 else SimpleTrainer()
    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    train()
