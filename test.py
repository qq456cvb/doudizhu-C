# from env import Env
import sys
sys.path.insert(0, './build/Release')
import env
# from env_test import Env
import card
from card import action_space, Category
import os, random
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import time
from env_test import get_benchmark
from collections import Counter
import struct
import copy
import random
import argparse

PASS_PENALTY = 5

##################################################### UTILITIES ########################################################
def counter_subset(list1, list2):
    c1, c2 = Counter(list1), Counter(list2)

    for (k, n) in c1.items():
        if n > c2[k]:
            return False
    return True

# map char cards to 3 - 17
def to_value(cards):
    values = [card.Card.cards.index(c)+3 for c in cards]
    return values

# map 3 - 17 to char cards
def to_char(cards):
    chars = [card.Card.cards[c-3] for c in cards]
    return chars

def get_mask(cards, action_space, last_cards):
    # 1 valid; 0 invalid
    mask = np.zeros([len(action_space)])
    for j in range(mask.size):
        if counter_subset(action_space[j], cards):
            mask[j] = 1
    mask = mask.astype(bool)
    if last_cards:
        for j in range(1, mask.size):
            if mask[j] == 1 and not card.CardGroup.to_cardgroup(action_space[j]).\
                    bigger_than(card.CardGroup.to_cardgroup(last_cards)):
                mask[j] = False
    # else:
    #     mask[0] = False
    return mask

# get char cards, return valid response
def get_mask_category(cards, action_space, last_cards=None):
    mask = np.zeros([14]) if last_cards is None else np.zeros([15])
    for i in range(action_space):
        if counter_subset(action_space[i], cards):
            if last_cards is None:
                mask[char2value_3_17(action_space[i][0])-3] = 1
            else:
                diff = char2value_3_17(action_space[i][0]) - char2value_3_17(last_cards[0])
                if diff > 0:
                    mask[diff-1] = 1
    return mask.astype(bool)

# get [-1, 1] minor cards target, input: value cards 3-17
def find_minor_in_three_one(cards):
    if cards[0] == cards[1]:
        return cards[-1]
    else:
        return cards[0]

def find_minor_in_three_two(cards):
    if cards[1] == cards[2]:
        return cards[-1]
    else:
        return cards[0]

def find_minor_in_three_one_line(cards):
    cnt = np.zeros([18])
    for i in range(len(cards)):
        cnt[cards[i]] += 1
    minor = []
    for i in range(3, 18):
        if cnt[i] == 1:
            minor.append(i)
    return np.array(minor)

def find_minor_in_three_two_line(cards):
    cnt = np.zeros([18])
    for i in range(len(cards)):
        cnt[cards[i]] += 1
    minor = []
    for i in range(3, 18):
        if cnt[i] == 2:
            minor.append(i)
    return np.array(minor)

def find_minor_in_four_two(cards):
    cnt = np.zeros([18])
    for i in range(len(cards)):
        cnt[cards[i]] += 1
    minor = []
    for i in range(3, 18):
        if cnt[i] == 1:
            minor.append(i)
    return np.array(minor)

def get_minor_cards(cards, category_idx):
    minor_cards = np.ones([15])
    length = 0
    if category_idx == Category.THREE_ONE.value:
        length = 1
        minor_cards[find_minor_in_three_one(cards)-3] = -1
    if category_idx == Category.THREE_TWO.value:
        length = 1
        minor_cards[find_minor_in_three_two(cards)-3] = -1
    if category_idx == Category.THREE_ONE_LINE.value:
        length = int(cards.size / 4)
        minor_cards[find_minor_in_three_one_line(cards)-3] = -1
    if category_idx == Category.THREE_TWO_LINE.value:
        length = int(cards.size / 5)
        minor_cards[find_minor_in_three_two_line(cards)-3] = -1
    if category_idx == Category.FOUR_TWO.value:
        length = 2
        minor_cards[find_minor_in_four_two(cards)-3] = -1
    return minor_cards, length

def update_params(scope_from, scope_to):
    vars_from = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_from)
    vars_to = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_to)

    ops = []
    for from_var, to_var in zip(vars_from, vars_to):
        ops.append(to_var.assign(from_var))
    return ops


def discounted_return(r, gamma):
    r = r.astype(float)
    r_out = np.zeros_like(r)
    val = 0
    for i in reversed(range(r.shape[0])):
        r_out[i] = r[i] + gamma * val
        val = r_out[i]
    return r_out

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def get_feature_state(env, mask=None):
    curr_cards = to_char(env.get_curr_cards())
    curr_val, curr_round = env.get_cards_value(card.Card.char2color(curr_cards))
    if mask is None:
        mask = get_mask(curr_cards, action_space, to_char(env.get_last_cards()))
    features = np.zeros([len(mask), 9])
    features[:, 0] = mask.astype(np.int32)
    for i in range(mask.shape[0]):
        m = mask[i]
        if m:
            a = action_space[i]
            
            if not a:
                features[i, 1] = 1
                continue
            next_cards = curr_cards.copy()
            for c in a:
                next_cards.remove(c)
            next_val, next_round = env.get_cards_value(card.Card.char2color(next_cards))
            lose_control = env.will_lose_control(card.Card.char2value_3_17(a) + 3)
            if lose_control:
                features[i, 1] = 1
            if len(a) >= len(curr_cards):
                features[i, 2] = 1
            if next_val > curr_val:
                features[i, 3] = 1
            if next_round < curr_round:
                features[i, 4] = 1

            cnt = len(a)
            if cnt > 15:
                cnt = 15
            features[i, 5] = cnt & 8 >> 3
            features[i, 6] = cnt & 4 >> 2
            features[i, 7] = cnt & 2 >> 1
            features[i, 8] = cnt & 1
    return features
    

##################################################### UTILITIES ########################################################
class CardNetwork:
    def __init__(self, s_dim, trainer, scope, a_dim=9085):
        with tf.variable_scope(scope):
            #card_cnt = 57
            #self.temp = tf.placeholder(tf.float32, None, name="boltz")
            with tf.name_scope("input_state"):
                self.input_state = tf.placeholder(tf.float32, [None, s_dim], name="input")
            with tf.name_scope("training"):
                self.training = tf.placeholder(tf.bool, None, name="mode")
            with tf.name_scope("input_single"):
                self.input_single = tf.placeholder(tf.float32, [None, 15], name="input_single")
            with tf.name_scope("input_pair"):
                self.input_pair = tf.placeholder(tf.float32, [None, 13], name="input_pair")
            with tf.name_scope("input_triple"):
                self.input_triple = tf.placeholder(tf.float32, [None, 13], name="input_triple")
            with tf.name_scope("input_quadric"):
                self.input_quadric = tf.placeholder(tf.float32, [None, 13], name="input_quadric")

            # TODO: test if embedding would help
            with tf.name_scope("input_state_embedding"):
                self.embeddings = slim.fully_connected(
                    inputs=self.input_state, 
                    num_outputs=512, 
                    activation_fn=tf.nn.elu,
                    weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("reshaping_for_conv"):
                self.input_state_conv = tf.reshape(self.embeddings, [-1, 1, 512, 1])
                self.input_single_conv = tf.reshape(self.input_single, [-1, 1, 15, 1])
                self.input_pair_conv = tf.reshape(self.input_pair, [-1, 1, 13, 1])
                self.input_triple_conv = tf.reshape(self.input_triple, [-1, 1, 13, 1])
                self.input_quadric_conv = tf.reshape(self.input_quadric, [-1, 1, 13, 1])

            # convolution for legacy state
            with tf.name_scope("conv_legacy_state"):
                self.state_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_state_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1a = tf.layers.batch_normalization(self.state_conv1a_branch1a, training=self.training)
                self.state_nonlinear1a_branch1a = tf.nn.relu(self.state_bn1a_branch1a)

                self.state_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.state_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1b = tf.layers.batch_normalization(self.state_conv1a_branch1b, training=self.training)
                self.state_nonlinear1a_branch1b = tf.nn.relu(self.state_bn1a_branch1b)

                self.state_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.state_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1c = tf.layers.batch_normalization(self.state_conv1a_branch1c, training=self.training)

                ######

                self.state_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_state_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch2 = tf.layers.batch_normalization(self.state_conv1a_branch2, training=self.training)

                self.state1a = self.state_bn1a_branch1c + self.state_bn1a_branch2
                self.state_output = slim.flatten(tf.nn.relu(self.state1a))

            # convolution for single
            with tf.name_scope("conv_single"):
                self.single_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_single_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1a = tf.layers.batch_normalization(self.single_conv1a_branch1a, training=self.training)
                self.single_nonlinear1a_branch1a = tf.nn.relu(self.single_bn1a_branch1a)

                self.single_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1b = tf.layers.batch_normalization(self.single_conv1a_branch1b, training=self.training)
                self.single_nonlinear1a_branch1b = tf.nn.relu(self.single_bn1a_branch1b)

                self.single_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1c = tf.layers.batch_normalization(self.single_conv1a_branch1c, training=self.training)

                ######

                self.single_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_single_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch2 = tf.layers.batch_normalization(self.single_conv1a_branch2, training=self.training)

                self.single1a = self.single_bn1a_branch1c + self.single_bn1a_branch2
                self.single_output = slim.flatten(tf.nn.relu(self.single1a))

            # convolution for pair
            with tf.name_scope("conv_pair"):
                self.pair_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1a = tf.layers.batch_normalization(self.pair_conv1a_branch1a, training=self.training)
                self.pair_nonlinear1a_branch1a = tf.nn.relu(self.pair_bn1a_branch1a)

                self.pair_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1b = tf.layers.batch_normalization(self.pair_conv1a_branch1b, training=self.training)
                self.pair_nonlinear1a_branch1b = tf.nn.relu(self.pair_bn1a_branch1b)

                self.pair_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1c = tf.layers.batch_normalization(self.pair_conv1a_branch1c, training=self.training)

                ######

                self.pair_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch2 = tf.layers.batch_normalization(self.pair_conv1a_branch2, training=self.training)

                self.pair1a = self.pair_bn1a_branch1c + self.pair_bn1a_branch2
                self.pair_output = slim.flatten(tf.nn.relu(self.pair1a))

            # convolution for triple
            with tf.name_scope("conv_triple"):
                self.triple_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1a = tf.layers.batch_normalization(self.triple_conv1a_branch1a, training=self.training)
                self.triple_nonlinear1a_branch1a = tf.nn.relu(self.triple_bn1a_branch1a)

                self.triple_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1b = tf.layers.batch_normalization(self.triple_conv1a_branch1b, training=self.training)
                self.triple_nonlinear1a_branch1b = tf.nn.relu(self.triple_bn1a_branch1b)

                self.triple_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1c = tf.layers.batch_normalization(self.triple_conv1a_branch1c, training=self.training)

                ######

                self.triple_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv, num_outputs=64,
                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch2 = tf.layers.batch_normalization(self.triple_conv1a_branch2, training=self.training)

                self.triple1a = self.triple_bn1a_branch1c + self.triple_bn1a_branch2
                self.triple_output = slim.flatten(tf.nn.relu(self.triple1a))

            # convolution for quadric
            with tf.name_scope("conv_quadric"):
                self.quadric_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1a = tf.layers.batch_normalization(self.quadric_conv1a_branch1a, training=self.training)
                self.quadric_nonlinear1a_branch1a = tf.nn.relu(self.quadric_bn1a_branch1a)

                self.quadric_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1b = tf.layers.batch_normalization(self.quadric_conv1a_branch1b, training=self.training)
                self.quadric_nonlinear1a_branch1b = tf.nn.relu(self.quadric_bn1a_branch1b)

                self.quadric_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1b, num_outputs=64,
                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1c = tf.layers.batch_normalization(self.quadric_conv1a_branch1c, training=self.training)

                ######

                self.quadric_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch2 = tf.layers.batch_normalization(self.quadric_conv1a_branch2, training=self.training)

                self.quadric1a = self.quadric_bn1a_branch1c + self.quadric_bn1a_branch2
                self.quadric_output = slim.flatten(tf.nn.relu(self.quadric1a))

            # 3 + 1 convolution
            with tf.name_scope("conv_3plus1"):
                tiled_triple = tf.tile(tf.expand_dims(self.input_triple, 1), [1, 15, 1])
                tiled_single = tf.tile(tf.expand_dims(self.input_single, 2), [1, 1, 13])
                self.input_triple_single_conv = tf.to_float(tf.expand_dims(tf.bitwise.bitwise_and(tf.to_int32(tiled_triple),
                    tf.to_int32(tiled_single)), -1))
                
                self.triple_single_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_triple_single_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1a = tf.layers.batch_normalization(self.triple_single_conv1a_branch1a, training=self.training)
                self.triple_single_nonlinear1a_branch1a = tf.nn.relu(self.triple_single_bn1a_branch1a)

                self.triple_single_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.triple_single_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[3, 3], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1b = tf.layers.batch_normalization(self.triple_single_conv1a_branch1b, training=self.training)
                self.triple_single_nonlinear1a_branch1b = tf.nn.relu(self.triple_single_bn1a_branch1b)

                self.triple_single_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.triple_single_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1c = tf.layers.batch_normalization(self.triple_single_conv1a_branch1c, training=self.training)

                ######

                self.triple_single_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_triple_single_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch2 = tf.layers.batch_normalization(self.triple_single_conv1a_branch2, training=self.training)

                self.triple_single1a = self.triple_single_bn1a_branch1c + self.triple_single_bn1a_branch2
                self.triple_single_output = slim.flatten(tf.nn.relu(self.triple_single1a))

            # 3 + 2 convolution
            with tf.name_scope("conv_3plus2"):
                tiled_triple = tf.tile(tf.expand_dims(self.input_triple, 1), [1, 13, 1])
                tiled_double = tf.tile(tf.expand_dims(self.input_pair, 2), [1, 1, 13])
                self.input_triple_double_conv = tf.to_float(tf.expand_dims(tf.bitwise.bitwise_and(tf.to_int32(tiled_triple),
                    tf.to_int32(tiled_double)), -1))
                
                self.triple_double_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_triple_double_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1a = tf.layers.batch_normalization(self.triple_double_conv1a_branch1a, training=self.training)
                self.triple_double_nonlinear1a_branch1a = tf.nn.relu(self.triple_double_bn1a_branch1a)
    
                self.triple_double_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.triple_double_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[3, 3], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1b = tf.layers.batch_normalization(self.triple_double_conv1a_branch1b, training=self.training)
                self.triple_double_nonlinear1a_branch1b = tf.nn.relu(self.triple_double_bn1a_branch1b)

                self.triple_double_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.triple_double_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1c = tf.layers.batch_normalization(self.triple_double_conv1a_branch1c, training=self.training)

                ######

                self.triple_double_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_triple_double_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch2 = tf.layers.batch_normalization(self.triple_double_conv1a_branch2, training=self.training)

                self.triple_double1a = self.triple_double_bn1a_branch1c + self.triple_double_bn1a_branch2
                self.triple_double_output = slim.flatten(tf.nn.relu(self.triple_double1a))

            #################################################

            with tf.name_scope("concated"):
                self.fc_flattened = tf.concat([self.single_output, self.pair_output, self.triple_output,
                    self.quadric_output, self.triple_single_output, self.triple_double_output, self.state_output], 1)

            # passive decision making  0: pass, 1: bomb, 2: king, 3: normal
            with tf.name_scope("passive_decision_making"):
                self.fc_decision_passive = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_decision_passive, slim.fully_connected, [128, 64, 32])
                self.fc_decision_passive_output = slim.fully_connected(self.fc_decision_passive, 4, tf.nn.softmax)

            # passive response
            with tf.name_scope("passive_response"):
                self.fc_response_passive = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_response_passive, slim.fully_connected, [128, 64, 32])
                self.fc_response_passive_output = slim.fully_connected(self.fc_response_passive, 14, tf.nn.softmax)

            # passive bomb response
            with tf.name_scope("passive_bomb_reponse"):
                self.fc_bomb_passive = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_bomb_passive, slim.fully_connected, [128, 64, 32])
                self.fc_bomb_passive_output = slim.fully_connected(self.fc_bomb_passive, 13, tf.nn.softmax)

            # active decision making  mapped to [action space category - 1]
            with tf.name_scope("active_decision_making"):
                self.fc_decision_active = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_decision_active, slim.fully_connected, [128, 64, 32])
                self.fc_decision_active_output = slim.fully_connected(self.fc_decision_active, 13, tf.nn.softmax)

            # active response
            with tf.name_scope("active_response"):
                self.fc_response_active = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_response_active, slim.fully_connected, [128, 64, 32])
                self.fc_response_active_output = slim.fully_connected(self.fc_response_active, 15, tf.nn.softmax)

            # minor card value map output [-1, 1]
            with tf.name_scope("fc_cards_value_output"):
                self.fc_cards_value = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_cards_value, slim.fully_connected, [128, 64, 32])
                self.fc_cards_value_output = slim.fully_connected(self.fc_cards_value, 15, tf.nn.tanh)

            # active or passive
            with tf.name_scope("input_is_active"):
                self.is_active = tf.placeholder(tf.bool, [None], name='active')

            # minor cards
            with tf.name_scope("minor_cards"):
                self.has_minor_cards = tf.placeholder(tf.bool, [None], name='minor_cards')
                self.minor_mask = tf.to_float(self.has_minor_cards)
                self.minor_cards_target = tf.placeholder(tf.float32, [None, 15], name='minor_cards_target')
                self.minor_loss = tf.reduce_sum(tf.square(self.minor_cards_target - self.fc_cards_value_output), 1)
                self.minor_loss = self.minor_mask * self.minor_loss

            # passive mode
            with tf.name_scope("passive_mode_loss"):
                self.is_passive_bomb = tf.placeholder(tf.bool, [None], name='passive_bomb')
                self.is_passive_king = tf.placeholder(tf.bool, [None], name='passive_is_king')

                self.passive_decision_input = tf.placeholder(tf.int32, [None], name='passive_decision_in')
                self.passive_decision_target = tf.one_hot(self.passive_decision_input, 4)
                self.passive_decision_loss = -tf.reduce_sum(self.passive_decision_target * tf.log(tf.clip_by_value(self.fc_decision_passive_output, 1e-10, 1-(1e-10))), 1)

                self.passive_response_input = tf.placeholder(tf.int32, [None], name='passive_response_in')
                self.passive_response_target = tf.one_hot(self.passive_response_input, 14)
                self.passive_response_loss = -tf.reduce_sum(self.passive_response_target * tf.log(tf.clip_by_value(self.fc_response_passive_output, 1e-10, 1-(1e-10))), 1)

                self.passive_bomb_input = tf.placeholder(tf.int32, [None], name='passive_bomb_in')
                self.passive_bomb_target = tf.one_hot(self.passive_bomb_input, 13)
                self.passive_bomb_loss = -tf.reduce_sum(self.passive_bomb_target * tf.log(tf.clip_by_value(self.fc_bomb_passive_output, 1e-10, 1-(1e-10))), 1)

            # active mode
            with tf.name_scope("active_mode_loss"):
                self.active_decision_input = tf.placeholder(tf.int32, [None], name='active_decision_in')
                self.active_decision_target = tf.one_hot(self.active_decision_input, 13)
                self.active_decision_loss = -tf.reduce_sum(self.active_decision_target * tf.log(tf.clip_by_value(self.fc_decision_active_output, 1e-10, 1-(1e-10))), 1)

                self.active_response_input = tf.placeholder(tf.int32, [None], name='active_response_in')
                self.active_response_target = tf.one_hot(self.active_response_input, 15)
                self.active_response_loss = -tf.reduce_sum(self.active_response_target * tf.log(tf.clip_by_value(self.fc_response_active_output, 1e-10, 1-(1e-10))), 1)


            with tf.name_scope("passive_loss"):
                self.passive_loss = tf.reduce_sum(self.passive_decision_loss + (1 - tf.to_float(self.is_passive_king)) * ((1 - tf.to_float(self.is_passive_bomb)) * self.passive_response_loss + \
                    tf.to_float(self.is_passive_bomb) * self.passive_bomb_loss))

            with tf.name_scope("active_loss"):
                self.active_loss = tf.reduce_sum(self.active_decision_loss + self.active_response_loss)

            with tf.name_scope("total_loss"):
                self.loss = tf.to_float(self.is_active) * self.active_loss + (1 - tf.to_float(self.is_active)) * self.passive_loss + self.minor_loss

            with tf.name_scope("optimize"):
                self.optimize = trainer.minimize(self.loss)

            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self.gradients = tf.gradients(self.loss, local_vars)

            # self.val_pred = tf.reshape(self.fc4, [-1])


class CardAgent:
    def __init__(self, name, trainer):
        self.name = name
        self.episodes = tf.Variable(0, dtype=tf.int32, name='episodes_' + name, trainable=False)
        self.increment = self.episodes.assign_add(1)
        self.network = CardNetwork(54 * 6, trainer, self.name, 8310)

    def train_batch_packed(self, buffer, masks, sess, gamma, val_last):
        states = buffer[:, 0]
        actions = buffer[:, 1]
        rewards = buffer[:, 2]
        values = buffer[:, 3]
        a_dims = buffer[:, 4]

        rewards_plus = np.append(rewards, val_last)
        val_truth = discounted_return(rewards_plus, gamma)[:-1]

        val_pred_plus = np.append(values, val_last)
        td0 = rewards + gamma * val_pred_plus[1:] - val_pred_plus[:-1]
        advantages = discounted_return(td0, gamma)

        for i in range(buffer.shape[0]):
            s = states[i]
            a = actions[i]
            r = rewards[i]
            v = values[i]
            a_dim = a_dims[i]
            v_truth = val_truth[i]
            advantage = advantages[i]
            buff = np.array([[s, a, r, v, a_dim]])
            m = masks[i:i+1]
            p_norm, a0, pred_prob, loss, policy_loss, val_loss, var_norms, grad_norms = \
                self.train_batch(buff, m, sess, gamma, [v_truth], [advantage])
        
        return p_norm / buffer.shape[0], a0 / buffer.shape[0], pred_prob / buffer.shape[0], loss / buffer.shape[0], \
            policy_loss / buffer.shape[0], val_loss / buffer.shape[0], var_norms / buffer.shape[0], grad_norms / buffer.shape[0]

    def train_batch(self, buffer, masks, sess, gamma, val_truth, advantages):
        states = buffer[:, 0]
        actions = buffer[:, 1]
        rewards = buffer[:, 2]
        values = buffer[:, 3]
        a_dims = buffer[0, 4]

        # rewards_plus = np.append(rewards, val_last)
        # val_truth = discounted_return(rewards_plus, gamma)[:-1]

        # print(val_truth)
        # input('continue')

        # val_pred_plus = np.append(values, val_last)
        # td0 = rewards + gamma * val_pred_plus[1:] - val_pred_plus[:-1]
        # advantages = discounted_return(td0, gamma)
        # advantages = val_truth

        _, p_norm, a0, action_one_hot, valid_policy, pred_prob, loss, policy_loss, val_loss, var_norms, grad_norms = sess.run([self.network.apply_grads,
            self.network.policy_norm,
            self.network.a0,
            self.network.action_one_hot,
            self.network.valid_policy,
            self.network.pred_prob,
            self.network.loss, 
            self.network.policy_loss, 
            self.network.val_loss,
            self.network.var_norms,
            self.network.grad_norms], 
            feed_dict={self.network.val_truth: val_truth,
                        self.network.advantages: advantages,
                        self.network.input: np.vstack(states),
                        self.network.action: actions,
                        self.network.masked_a_dim: a_dims,
                        self.network.mask: masks})
        # print("policy_norm:", p_norm, "a0", a0)
        # print(action_one_hot)
        # print(valid_policy)
        # print(val_truth)
        # print(val_loss)
        # input('continue')
        episode = sess.run(self.episodes)
        return p_norm, a0, pred_prob, loss, policy_loss, val_loss, var_norms, grad_norms
        # if episode % 100 == 0:
        #     print("loss : %f" % loss)

class CardMaster:
    def __init__(self, env):
        self.temp = 1
        self.start_temp = 1
        self.end_temp = 0.2
        self.action_space = card.get_action_space()
        self.name = 'global'
        self.env = env
        self.a_dim = 9085
        self.gamma = 0.99
        self.sess = None

        self.train_intervals = 30

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.episode_rewards = [[] for i in range(2)]
        self.episode_length = [[] for i in range(2)]
        self.episode_mean_values = [[] for i in range(2)]
        self.summary_writers = [tf.summary.FileWriter("train_agent%d" % i) for i in range(2)]

        self.agents = [CardAgent('agent%d' % i, self.trainer) for i in range(2)]

        self.global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        self.increment = self.global_episodes.assign_add(1)

    def train_batch(self, buffer, masks, sess, gamma, val_last, idx):
        buffer = np.array(buffer)
        masks = np.array(masks)
        return self.agents[idx].train_batch(buffer, masks, sess, gamma, val_last)

    def train_batch_packed(self, buffer, masks, sess, gamma, val_last, idx):
        buffer = np.array(buffer)
        masks = np.array(masks)
        return self.agents[idx].train_batch_packed(buffer, masks, sess, gamma, val_last)

    def respond(self, env):
        mask = get_mask(to_char(self.env.get_curr_cards()), self.action_space, to_char(self.env.get_last_cards()))
        s = env.get_state()
        s = np.reshape(s, [1, -1])
        policy, val = self.sess.run([
            self.agents[0].network.valid_policy,
            self.agents[0].network.val_pred],
            feed_dict={
                self.agents[0].network.input: s,
                self.agents[0].network.mask: np.reshape(mask, [1, -1])
            })
        policy = policy[0]
        valid_actions = np.take(np.arange(self.a_dim), mask.nonzero())
        valid_actions = valid_actions.reshape(-1)
        # a = np.random.choice(valid_actions, p=policy)
        a = valid_actions[np.argmax(policy)]
        # print("taking action: ", self.action_space[a])
        return env.step(self.action_space[a])

    # train two farmers simultaneously
    def run(self, sess, saver, max_episode_length, cards):
        self.sess = sess
        with sess.as_default():
            global_episodes = sess.run(self.global_episodes)
            total_episodes = 10001
            temp_decay = (self.end_temp - self.start_temp) / total_episodes
            while global_episodes < total_episodes:
                print("episode %d" % global_episodes)
                episode_buffer = [[] for i in range(2)]
                episode_mask = [[] for i in range(2)] 
                episode_values = [[] for i in range(2)]
                episode_reward = [0, 0]
                episode_steps = [0, 0]
                need_train = []

                self.env.reset()
                self.env.prepare()
                self.env.step(lord=True)

                # print("training id %d" % train_id)
                s = self.env.get_state()
                s = np.reshape(s, [1, -1])
                # rnn_state = [[c_init, h_init], [c_init.copy(), h_init.copy()]]
                # # shallow copy
                # rnn_state_backup = rnn_state.copy()
                for l in range(max_episode_length):
                    # time.sleep(1)
                    # # map 1, 3 to 0, 1
                    train_id = self.env.get_role_ID()
                    train_id = int((train_id - 1) / 2)

                    print("turn %d" % l)
                    print("training id %d" % train_id)
                    
                    mask = get_mask(to_char(self.env.get_curr_cards()), self.action_space, to_char(self.env.get_last_cards()))

                    # oppo_cnt = self.env.get_opponent_min_cnt()
                    # encourage response when opponent is about to win
                    # if random.random() > oppo_cnt / 20. and np.count_nonzero(mask) > 1:
                    #     mask[0] = False
                    policy, val = sess.run([
                        self.agents[train_id].network.boltz_policy,
                        self.agents[train_id].network.val_pred],
                        feed_dict={
                            self.agents[train_id].network.temp : self.temp,
                            self.agents[train_id].network.input: s,
                            self.agents[train_id].network.mask: np.reshape(mask, [1, -1])
                        })
                    self.temp -= temp_decay
                    policy = policy[0]
                    valid_actions = np.take(np.arange(self.a_dim), mask.nonzero())
                    valid_actions = valid_actions.reshape(-1)
                    a = np.random.choice(valid_actions, p=policy)

                    
                    a_masked = np.where(valid_actions == a)[0]


                    # print("taking action: ", self.action_space[a])
                    r, done = self.env.step(cards=to_value(self.action_space[a]))
                    s_prime = self.env.get_state()
                    s_prime = np.reshape(s_prime, [1, -1])

                    episode_buffer[train_id].append([s, a_masked, r, val[0], np.sum(mask.astype(np.float32))])
                    episode_mask[train_id].append(mask)
                    episode_values[train_id].append(val)
                    episode_reward[train_id] += r
                    episode_steps[train_id] += 1

                    if done:
                        for i in range(2):
                            if len(episode_buffer[i]) != 0:
                                p_norm, a0, pred_prob, loss, policy_loss, val_loss, var_norms, grad_norms = self.train_batch_packed(episode_buffer[i], episode_mask[i], sess, self.gamma, 0, i)
                        break
                        

                    s = s_prime

                    if len(episode_buffer[train_id]) == self.train_intervals:
                        val_last = sess.run(self.agents[train_id].network.val_pred,
                                            feed_dict={self.agents[train_id].network.input: s})
                        # print(val_last[0])
                        p_norm, a0, pred_prob, loss, policy_loss, val_loss, var_norms, grad_norms = self.train_batch_packed(episode_buffer[train_id], episode_mask[train_id], sess, self.gamma, val_last[0], train_id)
                        episode_buffer[train_id] = []
                        episode_mask[train_id] = []

                for i in range(2):
                    self.episode_mean_values[i].append(np.mean(episode_values[i]))
                    self.episode_length[i].append(episode_steps[i])
                    self.episode_rewards[i].append(episode_reward[i])

                    episodes = sess.run(self.agents[i].episodes)
                    sess.run(self.agents[i].increment)

                    update_rate = 5
                    if episodes % update_rate == 0 and episodes > 0:
                        mean_reward = np.mean(self.episode_rewards[i][-update_rate:])
                        mean_length = np.mean(self.episode_length[i][-update_rate:])
                        mean_value = np.mean(self.episode_mean_values[i][-update_rate:])

                        summary = tf.Summary()
                        summary.value.add(tag='Performance/rewards', simple_value=float(mean_reward))
                        summary.value.add(tag='Performance/length', simple_value=float(mean_length))
                        summary.value.add(tag='Performance/values', simple_value=float(mean_value))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(val_loss))
                        summary.value.add(tag='Losses/Prob pred', simple_value=float(pred_prob))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(policy_loss))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(grad_norms))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
                        summary.value.add(tag='Losses/Policy Norm', simple_value=float(p_norm))
                        summary.value.add(tag='Losses/a0', simple_value=float(a0))

                        self.summary_writers[i].add_summary(summary, episodes)
                        self.summary_writers[i].flush()

                global_episodes += 1
                sess.run(self.increment)
                # if global_episodes % 50 == 0:
                #     saver.save(sess, './model' + '/model-' + str(global_episodes) + '.cptk')
                #     print("Saved Model")

                # self.env.end()


def run_game(sess, network):
    max_episode_length = 100
    lord_win_rate = 0
    for i in range(100):
        network.env.reset()
        network.env.players[0].trainable = True
        lord_idx = 2
        network.env.players[2].is_human = True
        network.env.prepare(lord_idx)

        s = network.env.get_state(0)
        s = np.reshape(s, [1, -1])

        while True:
            policy, val = sess.run([network.agent.network.policy_pred, network.agent.network.val_pred],
                                   feed_dict={network.agent.network.input: s})
            mask = network.env.get_mask(0)
            valid_actions = np.take(np.arange(network.a_dim), mask.nonzero())
            valid_actions = valid_actions.reshape(-1)
            valid_p = np.take(policy[0], mask.nonzero())
            if np.count_nonzero(valid_p) == 0:
                valid_p = np.ones([valid_p.size]) / float(valid_p.size)
            else:
                valid_p = valid_p / np.sum(valid_p)
            valid_p = valid_p.reshape(-1)
            a = np.random.choice(valid_actions, p=valid_p)

            r, done = network.env.step(0, a)
            s_prime = network.env.get_state(0)
            s_prime = np.reshape(s_prime, [1, -1])

            if done:
                idx = network.env.check_winner()
                if idx == lord_idx:
                    lord_win_rate += 1
                print("winner is player %d" % idx)
                print("..............................")
                break
            s = s_prime
    print("lord winning rate: %f" % (lord_win_rate / 100.0))

if __name__ == '__main__':
    '''
    demoGames = []
    read_seq3("seq")
    numOfDemos = len(demoGames)

    # print("hc : ", demoGames[1].handcards)

    N = 200000
    f = open("data", "w")
    collect_data()
    f.close()
    '''

    parser = argparse.ArgumentParser(description='fight the lord feature vector')
    # parser.add_argument('--b', type=int, help='batch size', default=32)
    parser.add_argument('--epoches_train', type=int, help='num of epochs to train', default=1)
    parser.add_argument('--epoches_test', type=int, help='num of epochs to test', default=0)
    parser.add_argument('--train', type=bool, help='whether to train', default=True)

    args = parser.parse_args(sys.argv[1:])
    epoches_train = args.epoches_train
    epoches_test = args.epoches_test

    a_dim = len(action_space)

    SLNetwork = CardNetwork(54 * 6, tf.train.AdamOptimizer(learning_rate=0.0001), "SLNetwork")
    e = env.Env()
    TRAIN = args.train
    sess = tf.Session()
    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter('foo', sess.graph)
    passive_decision_acc = 0.
    passive_bomb_acc = 0.
    passive_response_acc = 0.
    active_decision_acc = 0.
    active_response_acc = 0.
    minor_cards_acc = 0.
    passive_decision_cnt = 0
    passive_bomb_cnt = 0
    passive_response_cnt = 0
    active_decision_cnt = 0
    active_response_cnt = 0
    minor_cards_cnt = 0

    action_space_single = action_space[1:16]
    action_space_pair = action_space[16:29]
    action_space_triple = action_space[29:42]
    action_space_quadric = action_space[42:55]
    # TODO: support batch training
    if TRAIN:
        sess.run(tf.global_variables_initializer())
        for i in range(epoches_train):
            e.reset()
            e.prepare()
            
            curr_cards_value = e.get_curr_cards()
            curr_cards_char = to_char(curr_cards_value)
            last_cards_value = e.get_last_cards()
            # print("curr_cards = ", curr_cards_value)
            # print("last_cards = ", last_cards_value)
            last_category_idx = -1
            # last_cards_char = to_char(last_cards_value)
            # mask = get_mask(curr_cards_char, action_space, last_cards_char)

            input_single = get_mask(curr_cards_char, action_space_single, None)
            input_pair = get_mask(curr_cards_char, action_space_pair, None)
            input_triple = get_mask(curr_cards_char, action_space_triple, None)
            input_quadric = get_mask(curr_cards_char, action_space_quadric, None)

            s = e.get_state()
            s = np.reshape(s, [1, -1])
            
            # s = get_feature_state(e, mask)
            r = 0
            while r == 0:
                active = True
                if last_cards_value.size > 0:
                    active = False

                has_minor_cards = np.array([False])
                minor_cards_target = np.ones([1, 15])
                minor_cards_length = np.ones([1])
                is_passive_bomb = np.array([False])
                is_passive_king = np.array([False])
                passive_decision_input = np.array([0])
                passive_response_input = np.array([0])
                passive_bomb_input = np.array([0])
                active_decision_input = np.array([0])
                active_response_input = np.array([0])

                # if active:
                #     print("No one respond", end=':')
                # else:
                #     print("Another one respond", end=':')
                intention, r, category_idx = e.step_auto()
                #print("intention = ", intention)
                #print("r = ", r)
                #print("category_idx = ",category_idx)
                # print(intention)
                # do not account for 4 + 2 + 2 now
                if category_idx == 14:
                    curr_cards_value = e.get_curr_cards()
                    curr_cards_char = to_char(curr_cards_value)
                    last_cards_value = e.get_last_cards()
                    if last_cards_value.size > 0:
                        last_category_idx = category_idx
                    # last_cards_char = to_char(last_cards_value)
                    # mask = get_mask(curr_cards_char, action_space, last_cards_char)

                    input_single = get_mask(curr_cards_char, action_space_single, None)
                    input_pair = get_mask(curr_cards_char, action_space_pair, None)
                    input_triple = get_mask(curr_cards_char, action_space_triple, None)
                    input_quadric = get_mask(curr_cards_char, action_space_quadric, None)
                    continue

                if category_idx == Category.THREE_ONE.value or category_idx == Category.THREE_TWO.value or \
                        category_idx == Category.THREE_ONE_LINE.value or category_idx == Category.THREE_TWO_LINE.value or \
                        category_idx == Category.FOUR_TWO.value:
                    has_minor_cards[0] = True
                    minor_cards_target[0], minor_cards_length[0] = get_minor_cards(intention, category_idx)
                
                if not active:
                    if category_idx == Category.QUADRIC.value and category_idx != last_category_idx:
                        is_passive_bomb[0] = True
                        passive_decision_input[0] = 1
                        passive_bomb_input[0] = intention[0] - 3
                    else:
                        if category_idx == Category.BIGBANG.value:
                            is_passive_king[0] = True
                            passive_decision_input[0] = 2
                        else:
                            if category_idx != Category.EMPTY.value:
                                passive_decision_input[0] = 3
                                passive_response_input[0] = intention[0] - last_cards_value[0]
                else:
                    # ACTIVE OFFSET ONE!
                    active_decision_input[0] = category_idx - 1
                    active_response_input[0] = intention[0] - 3
                    

                _, decision_passive_output, response_passive_output, bomb_passive_output, \
                    decision_active_output, response_active_output, minor_cards_output, loss, \
                    active_decision_loss, active_response_loss, passive_decision_loss, passive_response_loss, passive_bomb_loss \
                     = sess.run([SLNetwork.optimize, SLNetwork.fc_decision_passive_output, 
                                SLNetwork.fc_response_passive_output, SLNetwork.fc_bomb_passive_output,
                                SLNetwork.fc_decision_active_output, SLNetwork.fc_response_active_output, 
                                SLNetwork.fc_cards_value_output, SLNetwork.loss,
                                SLNetwork.active_decision_loss, SLNetwork.active_response_loss, SLNetwork.passive_decision_loss, SLNetwork.passive_response_loss, SLNetwork.passive_bomb_loss],
                        feed_dict = {
                            SLNetwork.training: True,
                            SLNetwork.input_state: s,
                            SLNetwork.input_single: np.reshape(input_single, [1, -1]),
                            SLNetwork.input_pair: np.reshape(input_pair, [1, -1]),
                            SLNetwork.input_triple: np.reshape(input_triple, [1, -1]),
                            SLNetwork.input_quadric: np.reshape(input_quadric, [1, -1]),
                            SLNetwork.is_active: np.array([active]),
                            SLNetwork.has_minor_cards: has_minor_cards,
                            SLNetwork.minor_cards_target: minor_cards_target,
                            SLNetwork.is_passive_bomb: is_passive_bomb,
                            SLNetwork.is_passive_king: is_passive_king,
                            SLNetwork.passive_decision_input: passive_decision_input,
                            SLNetwork.passive_response_input: passive_response_input,
                            SLNetwork.passive_bomb_input: passive_bomb_input,
                            SLNetwork.active_decision_input: active_decision_input,
                            SLNetwork.active_response_input: active_response_input
                })

                #print("gradients ： ", gradients)

                # update accuracies
                if has_minor_cards[0]:
                    # print("minor_cards_output : ", minor_cards_output)
                    # print("minor_cards_output.argsort : ", minor_cards_output.argsort()[:int(minor_cards_length[0])])
                    # print("minor_cards_target[0] : ", minor_cards_target[0])
                    # print("minor_cards_target[0].argsort : ", minor_cards_target[0].argsort()[:int(minor_cards_length[0])])
                    minor_cards_acc_temp = 1 if np.array_equal(minor_cards_output.argsort()[:int(minor_cards_length[0])][0], \
                        minor_cards_target.argsort()[:int(minor_cards_length[0])][0]) else 0
                    minor_cards_cnt += 1
                    minor_cards_acc += (minor_cards_acc_temp - minor_cards_acc) / minor_cards_cnt
                if active:
                    active_decision_acc_temp = 1 if np.argmax(decision_active_output) == active_decision_input[0] else 0
                    active_decision_cnt += 1
                    active_decision_acc += (active_decision_acc_temp - active_decision_acc) / active_decision_cnt

                    active_response_acc_temp = 1 if np.argmax(response_active_output) == active_response_input[0] else 0
                    active_response_cnt += 1
                    active_response_acc += (active_response_acc_temp - active_response_acc) / active_response_cnt
                else:
                    passive_decision_acc_temp = 1 if np.argmax(decision_passive_output) == passive_decision_input[0] else 0
                    passive_decision_cnt += 1
                    passive_decision_acc += (passive_decision_acc_temp - passive_decision_acc) / passive_decision_cnt

                    if is_passive_bomb:
                        passive_bomb_acc_temp = 1 if np.argmax(bomb_passive_output) == passive_bomb_input[0] else 0
                        passive_bomb_cnt += 1
                        passive_bomb_acc += (passive_bomb_acc_temp - passive_bomb_acc) / passive_bomb_cnt
                    else:
                        passive_response_acc_temp = 1 if np.argmax(response_passive_output) == passive_response_input[0] else 0
                        passive_response_cnt += 1
                        passive_response_acc += (passive_response_acc_temp - passive_response_acc) / passive_response_cnt

                curr_cards_value = e.get_curr_cards()
                curr_cards_char = to_char(curr_cards_value)
                last_cards_value = e.get_last_cards()
                # print("curr_cards = ", curr_cards_value)
                # print("last_cards = ", last_cards_value)
                if last_cards_value.size > 0:
                    last_category_idx = category_idx
                    last_cards_char = to_char(last_cards_value)
                # mask = get_mask(curr_cards_char, action_space, last_cards_char)

                input_single = get_mask(curr_cards_char, action_space_single, last_cards_char if last_cards_value.size > 0 else None)
                input_pair = get_mask(curr_cards_char, action_space_pair, last_cards_char if last_cards_value.size > 0 else None)
                input_triple = get_mask(curr_cards_char, action_space_triple, last_cards_char if last_cards_value.size > 0 else None)
                input_quadric = get_mask(curr_cards_char, action_space_quadric, last_cards_char if last_cards_value.size > 0 else None)

                s = e.get_state()
                s = np.reshape(s, [1, -1])
            #print("End of one game")
            if i % 100 == 0:
                print("train1 ", i, " ing...")
                print("train passive decision accuracy = ", passive_decision_acc)
                print("train passive response accuracy = ", passive_response_acc)
                print("train passive bomb accuracy = ", passive_bomb_acc)
                print("train active decision accuracy = ", active_decision_acc)
                print("train active response accuracy = ", active_response_acc)
                print("train minor cards accuracy = ", minor_cards_acc)
                summary = tf.Summary(value=[
                                tf.Summary.Value(tag="Accuracy_passive_decision_accuracy", simple_value=passive_decision_acc), 
                                tf.Summary.Value(tag="Accuracy_passive_response_accuracy", simple_value=passive_response_acc),
                                tf.Summary.Value(tag="Accuracy_passive_bomb_accuracy", simple_value=passive_bomb_acc),
                                tf.Summary.Value(tag="Accuracy_active_decision_accuracy", simple_value=active_decision_acc),
                                tf.Summary.Value(tag="Accuracy_active_response_accuracy", simple_value=active_response_acc),
                                tf.Summary.Value(tag="Accuracy_minor_cards_accuracy", simple_value=minor_cards_acc),
                                #tf.Summary.Value(tag="minor_loss", simple_value=minor_loss)
                                tf.Summary.Value(tag="Loss_loss", simple_value=loss[0]),
                                tf.Summary.Value(tag="Loss_active_decision_loss", simple_value=active_decision_loss[0]),
                                tf.Summary.Value(tag="Loss_active_response_loss", simple_value=active_response_loss[0]),
                                tf.Summary.Value(tag="Loss_passive_decision_loss", simple_value=passive_decision_loss[0]),
                                tf.Summary.Value(tag="Loss_passive_response_loss", simple_value=passive_response_loss[0]),
                                tf.Summary.Value(tag="Loss_passive_bomb_loss", simple_value=passive_bomb_loss[0]) 
                                #tf.Summary.Value(tag="gradients", )
                ])
                file_writer.add_summary(summary, i / 100 - 1)
                #if i % 1000 == 0:
                    #saver.save(sess, "./foom/accuracy_bugfixed_lr0001_clipped/model.ckpt")

        
        # saver.save(sess, "./Model/SLNetwork_feat_deeper_1000000epoches.ckpt")

    print("train passive decision accuracy = ", passive_decision_acc)
    print("train passive response accuracy = ", passive_response_acc)
    print("train passive bomb accuracy = ", passive_bomb_acc)
    print("train active decision accuracy = ", active_decision_acc)
    print("train active response accuracy = ", active_response_acc)
    print("train minor cards accuracy = ", minor_cards_acc)

    file_writer.close()

    # test part
    # saver.restore(sess, "./Model/SLNetwork_feat_deeper_1000000epoches.ckpt")

    cnt = 0
    acc = 0.
    for i in range(epoches_test):
        e.reset()
        e.prepare()

        mask = get_mask(to_char(e.get_curr_cards()), action_space, to_char(e.get_last_cards()))
        s = get_feature_state(e, mask)
        r = 0
        while r == 0:
            intention, r, category_idx = e.step_auto()
            # no training for empty actions
            if np.count_nonzero(mask) == 1:
                mask = get_mask(to_char(e.get_curr_cards()), action_space, to_char(e.get_last_cards()))
                s = get_feature_state(e, mask)
                continue
            put_list = card.Card.to_cards_from_3_17(intention)
            print(put_list)
            
            try:
                a = next(i for i, v in enumerate(action_space) if v == put_list)
            except StopIteration as e:
                print(put_list)

            valid_policy = sess.run(SLNetwork.valid_policy,
                                    feed_dict = {
                                        SLNetwork.training: False,
                                        SLNetwork.input: np.reshape(s, [1, -1]),
                                        SLNetwork.mask: np.reshape(mask, [1, -1])
                                    })
            valid_policy = valid_policy[0]
            valid_actions = np.take(np.arange(a_dim), np.array(mask).nonzero()).reshape(-1)
            # a = np.random.choice(valid_actions, p=valid_policy)
            a_pred = valid_actions[np.argmax(valid_policy)]
            accuracy = 0.
            if a == a_pred:
                accuracy = 1.
            cnt += 1
            acc += (accuracy - acc) / cnt

            mask = get_mask(to_char(e.get_curr_cards()), action_space, to_char(e.get_last_cards()))
            s = get_feature_state(e, mask)
        
        if i % 1000 == 0:
            print("predict ", i, " ing...")

    print("test accuracy = ", acc)
    sess.close()

