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
from TensorPack.ResNetBlock import identity_block, upsample_block, downsample_block
import tensorflow as tf

INPUT_DIM = 60 * 3
LAST_INPUT_DIM = 60
WEIGHT_DECAY = 5 * 1e-4
SCOPE = 'SL_policy_network'

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
    env = Env(rng.randint(1 << 31))
    # logger.info('called')

    while True:
        env.reset()
        env.prepare()
        r = 0
        while r == 0:
            last_cards_value = env.get_last_outcards()
            last_cards_char = to_char(last_cards_value)
            last_out_cards = Card.val2onehot60(last_cards_value)
            last_category_idx = env.get_last_outcategory_idx()
            curr_cards_char = to_char(env.get_curr_handcards())
            is_active = True if last_cards_value.size == 0 else False

            s = env.get_state_prob()
            # s = s[:60]
            intention, r, category_idx = env.step_auto()

            if category_idx == 14:
                continue
            minor_cards_targets = pick_minor_targets(category_idx, to_char(intention))
            # self, state, last_cards, passive_decision_target, passive_bomb_target, passive_response_target,
            # active_decision_target, active_response_target, seq_length_target, minor_response_target, minor_type, mode
            if not is_active:
                if category_idx == Category.QUADRIC.value and category_idx != last_category_idx:
                    passive_decision_input = 1
                    passive_bomb_input = intention[0] - 3
                    yield s, last_out_cards, passive_decision_input, 0, 0, 0, 0, 0, 0, 0, 0
                    yield s, last_out_cards, 0, passive_bomb_input, 0, 0, 0, 0, 0, 0, 1

                else:
                    if category_idx == Category.BIGBANG.value:
                        passive_decision_input = 2
                        yield s, last_out_cards, passive_decision_input, 0, 0, 0, 0, 0, 0, 0, 0
                    else:
                        if category_idx != Category.EMPTY.value:
                            passive_decision_input = 3
                            # OFFSET_ONE
                            # 1st, Feb - remove relative card output since shift is hard for the network to learn
                            passive_response_input = intention[0] - 3
                            if passive_response_input < 0:
                                print("something bad happens")
                                passive_response_input = 0
                            yield s, last_out_cards, passive_decision_input, 0, 0, 0, 0, 0, 0, 0, 0
                            yield s, last_out_cards, 0, 0, passive_response_input, 0, 0, 0, 0, 0, 2
                        else:
                            passive_decision_input = 0
                            yield s, last_out_cards, passive_decision_input, 0, 0, 0, 0, 0, 0, 0, 0

            else:
                seq_length = get_seq_length(category_idx, intention)

                # ACTIVE OFFSET ONE!
                active_decision_input = category_idx - 1
                active_response_input = intention[0] - 3
                yield s, last_out_cards, 0, 0, 0, active_decision_input, 0, 0, 0, 0, 3
                yield s, last_out_cards, 0, 0, 0, 0, active_response_input, 0, 0, 0, 4

                if seq_length is not None:
                    # length offset one
                    seq_length_input = seq_length - 1
                    yield s, last_out_cards, 0, 0, 0, 0, 0, seq_length_input, 0, 0, 5

            if minor_cards_targets is not None:
                main_cards = pick_main_cards(category_idx, to_char(intention))
                handcards = curr_cards_char.copy()
                state = s.copy()
                for main_card in main_cards:
                    handcards.remove(main_card)
                cards_onehot = Card.char2onehot60(main_cards)

                # we must make the order in each 4 batch correct...
                discard_onehot_from_s_60(state, cards_onehot)

                is_pair = False
                minor_type = 0
                if category_idx == Category.THREE_TWO.value or category_idx == Category.THREE_TWO_LINE.value:
                    is_pair = True
                    minor_type = 1
                for target in minor_cards_targets:
                    target_val = Card.char2value_3_17(target) - 3
                    yield state.copy(), last_out_cards, 0, 0, 0, 0, 0, 0, target_val, minor_type, 6
                    cards = [target]
                    handcards.remove(target)
                    if is_pair:
                        if target not in handcards:
                            print('something wrong...')
                            print('minor', target)
                            print('main_cards', main_cards)
                            print('handcards', handcards)
                            print('intention', intention)
                            print('category_idx', category_idx)
                        else:
                            handcards.remove(target)
                            cards.append(target)

                    # correct for one-hot state
                    cards_onehot = Card.char2onehot60(cards)

                    # print(s.shape)
                    # print(cards_onehot.shape)
                    discard_onehot_from_s_60(state, cards_onehot)


class Model(ModelDesc):
    def get_pred(self, state, last_cards, minor_type):
        with tf.variable_scope(SCOPE):
            with slim.arg_scope([slim.fully_connected, slim.conv2d], weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
                with tf.variable_scope('branch_main'):
                    flattened_1 = conv_block(state[:, :60], 32, INPUT_DIM // 3, [[16, 32, 5, 'identity'],
                                                                      [16, 32, 5, 'identity'],
                                                                      [32, 128, 5, 'downsampling'],
                                                                      [32, 128, 5, 'identity'],
                                                                      [32, 128, 5, 'identity'],
                                                                      [64, 256, 5, 'downsampling'],
                                                                      [64, 256, 3, 'identity'],
                                                                      [64, 256, 3, 'identity']
                                                                      ], 'branch_main1')
                    flattened_2 = conv_block(state[:, 60:120], 32, INPUT_DIM // 3, [[16, 32, 5, 'identity'],
                                                                      [16, 32, 5, 'identity'],
                                                                      [32, 128, 5, 'downsampling'],
                                                                      [32, 128, 5, 'identity'],
                                                                      [32, 128, 5, 'identity'],
                                                                      [64, 256, 5, 'downsampling'],
                                                                      [64, 256, 3, 'identity'],
                                                                      [64, 256, 3, 'identity']
                                                                      ], 'branch_main2')
                    flattened_3 = conv_block(state[:, 120:], 32, INPUT_DIM // 3, [[16, 32, 5, 'identity'],
                                                                      [16, 32, 5, 'identity'],
                                                                      [32, 128, 5, 'downsampling'],
                                                                      [32, 128, 5, 'identity'],
                                                                      [32, 128, 5, 'identity'],
                                                                      [64, 256, 5, 'downsampling'],
                                                                      [64, 256, 3, 'identity'],
                                                                      [64, 256, 3, 'identity']
                                                                      ], 'branch_main3')
                    flattened = tf.concat([flattened_1, flattened_2, flattened_3], axis=1)

                with tf.variable_scope('branch_passive'):
                    flattened_last = conv_block(last_cards, 32, LAST_INPUT_DIM, [[16, 32, 5, 'identity'],
                                                                             [16, 32, 5, 'identity'],
                                                                             [32, 128, 5, 'downsampling'],
                                                                             [32, 128, 5, 'identity'],
                                                                             [32, 128, 5, 'identity'],
                                                                             [64, 256, 5, 'downsampling'],
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
                                                                                              sequence_length=tf.ones([
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
                                                                                              sequence_length=tf.ones([
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
        return [tf.placeholder(tf.float32, [None, INPUT_DIM], 'state_in'),
                tf.placeholder(tf.float32, [None, LAST_INPUT_DIM], 'last_cards_in'),
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

    def build_graph(self, state, last_cards, passive_decision_target, passive_bomb_target, passive_response_target,
                    active_decision_target, active_response_target, seq_length_target, minor_response_target, minor_type, mode):
        (passive_decision_logits, passive_bomb_logits, passive_response_logits, active_decision_logits,
         active_response_logits, active_seq_logits, minor_response_logits) = self.get_pred(state, last_cards, minor_type)
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
        # vars = tf.trainable_variables()
        # for v in vars:
        #     print(v.name)
        # l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=SCOPE)
        # NOTE: this collection doesn't always grow with towers.
        # It only grows with actual variable creation, but not get_variable call.
        ctx = get_current_tower_context()
        if ctx.has_own_variables:  # be careful of the first tower (name='')
            l2_loss = ctx.get_collection_in_tower(tf.GraphKeys.REGULARIZATION_LOSSES)
        else:
            l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(l2_loss) > 0:
            logger.info("regularize_cost_from_collection() found {} regularizers "
                        "in REGULARIZATION_LOSSES collection.".format(len(l2_loss)))

        l2_main_loss = [l for l in l2_loss if 'branch_main' in l.name]
        l2_passive_fc_loss = [l for l in l2_loss if
                              'branch_passive' in l.name and 'decision' not in l.name and 'bomb' not in l.name and 'response' not in l.name]
        l2_active_fc_loss = [l for l in l2_loss if
                             'branch_active' in l.name and 'decision' not in l.name and 'response' not in l.name and 'seq_length' not in l.name] \
                            + [WEIGHT_DECAY * tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name(
            SCOPE + '/branch_active/decision/rnn/basic_lstm_cell/kernel:0'))]

        print('l2 loss', len(l2_loss))
        print('l2 main loss', len(l2_main_loss))
        print('l2 passive fc loss', len(l2_passive_fc_loss))
        print('l2 active fc loss', len(l2_active_fc_loss))

        name_scopes = ['branch_passive/decision', 'branch_passive/bomb', 'branch_passive/response',
                       'branch_active/decision', 'branch_active/response', 'branch_active/seq_length',
                       'branch_minor']

        # B * 7
        losses = [passive_decision_loss, passive_bomb_loss, passive_response_loss,
                  active_decision_loss, active_response_loss, active_seq_loss, minor_loss]

        for i, name in enumerate(name_scopes):
            l2_branch_loss = l2_main_loss.copy()
            if 'passive' in name:
                if 'bomb' in name:
                    l2_branch_loss += [l for l in l2_loss if name in l.name]
                else:
                    l2_branch_loss += l2_passive_fc_loss + [l for l in l2_loss if name in l.name]
            else:
                if 'minor' in name:
                    # do not include lstm regularization in minor loss
                    l2_branch_loss += l2_active_fc_loss[:-1] + [l for l in l2_loss if name in l.name]
                else:
                    l2_branch_loss += l2_active_fc_loss + [l for l in l2_loss if name in l.name]

            losses[i] += tf.add_n(l2_branch_loss)
            print('losses shape', losses[i].shape)
            print('l2 branch loss', len(l2_branch_loss))

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
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.3))]
                     # SummaryGradient()]
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
        train_tower = [0], [0]

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
            # ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            # ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            # HumanHyperParamSetter('learning_rate'),
            # HumanHyperParamSetter('entropy_beta')
            PeriodicTrigger(Evaluator(
                100, ['state_in', 'last_cards_in', 'minor_type_in'],
                ['passive_decision_prob', 'passive_bomb_prob', 'passive_response_prob',
                 'active_decision_prob', 'active_response_prob', 'active_seq_prob', 'minor_response_prob'], get_player),
                every_k_epochs=1),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=100,
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
            # print(intention)