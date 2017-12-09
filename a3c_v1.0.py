# from env import Env
import sys

sys.path.insert(0, './build/Release')
import env
# from env_test import Env
import card
from card import action_space, Category, action_space_category
import os, random
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import time
# from env_test import get_benchmark
from collections import Counter
import struct
import copy
import random
import argparse

action_space_single = action_space[1:16]
action_space_pair = action_space[16:29]
action_space_triple = action_space[29:42]
action_space_quadric = action_space[42:55]


##################################################### UTILITIES ########################################################
def counter_subset(list1, list2):
    c1, c2 = Counter(list1), Counter(list2)

    for (k, n) in c1.items():
        if n > c2[k]:
            return False
    return True


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm


# map char cards to 3 - 17
def to_value(cards):
    values = [card.Card.cards.index(c) + 3 for c in cards]
    return values


# map 3 - 17 to char cards
def to_char(cards):
    chars = [card.Card.cards[c - 3] for c in cards]
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
            if mask[j] == 1 and not card.CardGroup.to_cardgroup(action_space[j]). \
                    bigger_than(card.CardGroup.to_cardgroup(last_cards)):
                mask[j] = False
    # else:
    #     mask[0] = False
    return mask


# return <decision, response, minor cards> masks
def get_mask_alter(cards, last_cards, is_bomb, last_cards_category):
    decision_mask = None
    response_mask = None
    bomb_mask = None
    if len(last_cards) == 0:
        decision_mask = np.zeros([13])
        response_mask = np.zeros([13, 15])
        for i in range(13):
            # OFFSET ONE
            subspace = action_space_category[i + 1]
            for j in range(len(subspace)):
                if counter_subset(subspace[j], cards):
                    response_mask[i][card.Card.char2value_3_17(subspace[j][0]) - 3] = 1
                    decision_mask[i] = 1
        return decision_mask, response_mask, bomb_mask
    else:
        decision_mask = np.ones([4])
        decision_mask[3] = 0
        if not counter_subset(['*', '$'], cards):
            decision_mask[2] = 0
        if is_bomb:
            decision_mask[1] = 0
        response_mask = np.zeros([14])
        subspace = action_space_category[last_cards_category]
        for j in range(len(subspace)):
            if counter_subset(subspace[j], cards) and card.CardGroup.to_cardgroup(subspace[j]).\
                    bigger_than(card.CardGroup.to_cardgroup(last_cards)):
                diff = card.Card.to_value(subspace[j][0]) - card.Card.to_value(last_cards[0])
                assert(diff > 0)
                response_mask[diff - 1] = 1
                decision_mask[3] = 1
        if not is_bomb:
            bomb_mask = np.zeros([13])
            subspace = action_space_category[Category.QUADRIC.value]
            no_bomb = True
            for j in range(len(subspace)):
                if counter_subset(subspace[j], cards):
                    bomb_mask[card.Card.char2value_3_17(subspace[j][0]) - 3] = 1
                    no_bomb = False
            # if we got no bomb, we cannot respond with bombs
            if no_bomb:
                decision_mask[1] = 0
        return decision_mask, response_mask, bomb_mask


# return [3-17 value]
def give_cards_with_minor(response, actions_minor, hand_cards_value, last_cards_value, category_idx, length_output):
    # these mask will be used to tease out invalid card combinations
    single_mask = np.zeros([15])
    for i in range(3, 18):
        if i in hand_cards_value:
            single_mask[i - 3] = 1

    double_mask = np.zeros([13])
    for i in range(3, 16):
        if counter_subset([i, i], hand_cards_value):
            double_mask[i - 3] = 1

    if last_cards_value.size > 0:
        if category_idx == Category.SINGLE.value:
            return np.array([last_cards_value[0] + response])
        elif category_idx == Category.DOUBLE.value:
            return np.array([last_cards_value[0] + response] * 2)
        elif category_idx == Category.TRIPLE.value:
            return np.array([last_cards_value[0] + response] * 3)
        elif category_idx == Category.QUADRIC.value:
            return np.array([last_cards_value[0] + response] * 4)
        elif category_idx == Category.THREE_ONE.value:
            single_mask[last_cards_value[0] + response - 3] = 0
            return np.array([last_cards_value[0] + response] * 3 + [actions_minor[0] + 3])
        elif category_idx == Category.THREE_TWO.value:
            double_mask[last_cards_value[0] + response - 3] = 0
            return np.array([last_cards_value[0] + response] * 3 + [actions_minor[0] + 3] * 2)
        elif category_idx == Category.SINGLE_LINE.value:
            return np.arange(last_cards_value[0] + response, last_cards_value[0] + response + len(last_cards_value))
        elif category_idx == Category.DOUBLE_LINE.value:
            link = np.arange(last_cards_value[0] + response,
                             last_cards_value[0] + response + int(len(last_cards_value) / 2))
            return np.array([link, link]).T.reshape(-1)
        elif category_idx == Category.TRIPLE_LINE.value:
            link = np.arange(last_cards_value[0] + response,
                             last_cards_value[0] + response + int(len(last_cards_value) / 3))
            return np.array([link, link, link]).T.reshape(-1)
        elif category_idx == Category.THREE_ONE_LINE.value:
            cnt = int(len(last_cards_value) / 4)
            for j in range(last_cards_value[0] + response, last_cards_value[0] + response + cnt):
                single_mask[j - 3] = 0
            link = np.arange(last_cards_value[0] + response, last_cards_value[0] + response + cnt)
            main = np.array([link, link, link]).T.reshape(-1)
            # 1 dimension
            minor = actions_minor + 3
            return np.concatenate([main, minor])
        elif category_idx == Category.THREE_TWO_LINE.value:
            cnt = int(len(last_cards_value) / 5)
            for j in range(last_cards_value[0] + response, last_cards_value[0] + response + cnt):
                double_mask[j - 3] = 0
            link = np.arange(last_cards_value[0] + response, last_cards_value[0] + response + cnt)
            main = np.array([link, link, link]).T.reshape(-1)
            minor = actions_minor + 3
            minor = np.array([minor, minor]).T.reshape(-1)
            return np.concatenate([main, minor])
        elif category_idx == Category.FOUR_TWO.value:
            single_mask[last_cards_value[0] + response - 3] = 0
            minor = actions_minor + 3
            return np.array([last_cards_value[0] + response] * 4 + [minor[0]] + [minor[1]])
    else:
        if category_idx == Category.SINGLE.value:
            return np.array([response + 3])
        elif category_idx == Category.DOUBLE.value:
            return np.array([response + 3] * 2)
        elif category_idx == Category.TRIPLE.value:
            return np.array([response + 3] * 3)
        elif category_idx == Category.QUADRIC.value:
            return np.array([response + 3] * 4)
        elif category_idx == Category.THREE_ONE.value:
            single_mask[response] = 0
            return np.array([response + 3] * 3 + [actions_minor[0] + 3])
        elif category_idx == Category.THREE_TWO.value:
            double_mask[response] = 0
            return np.array([response + 3] * 3 + [actions_minor[0] + 3] * 2)
        elif category_idx == Category.SINGLE_LINE.value:
            # length output will be in range 1-12
            return np.arange(response + 3, response + 3 + length_output)
        elif category_idx == Category.DOUBLE_LINE.value:
            link = np.arange(response + 3, response + 3 + length_output)
            return np.array([link, link]).T.reshape(-1)
        elif category_idx == Category.TRIPLE_LINE.value:
            link = np.arange(response + 3, response + 3 + length_output)
            return np.array([link, link, link]).T.reshape(-1)
        elif category_idx == Category.THREE_ONE_LINE.value:
            cnt = length_output
            for j in range(response + 3, response + 3 + cnt):
                single_mask[j - 3] = 0
            link = np.arange(response + 3, response + 3 + cnt)
            main = np.array([link, link, link]).T.reshape(-1)
            minor = actions_minor + 3
            return np.concatenate([main, minor])
        elif category_idx == Category.THREE_TWO_LINE.value:
            cnt = length_output
            for j in range(response + 3, response + 3 + cnt):
                double_mask[j - 3] = 0
            link = np.arange(response + 3, response + 3 + cnt)
            main = np.array([link, link, link]).T.reshape(-1)
            minor = actions_minor + 3
            minor = np.array([minor, minor]).T.reshape(-1)
            return np.concatenate([main, minor])
        elif category_idx == Category.FOUR_TWO.value:
            single_mask[response] = 0
            minor = actions_minor + 3
            return np.array([response + 3] * 4 + [minor[0]] + [minor[1]])


# get char cards, return valid response
# def get_mask_category(cards, action_space, last_cards=None):
#     mask = np.zeros([14]) if last_cards is None else np.zeros([15])
#     for i in range(action_space):
#         if counter_subset(action_space[i], cards):
#             if last_cards is None:
#                 mask[char2value_3_17(action_space[i][0]) - 3] = 1
#             else:
#                 diff = char2value_3_17(action_space[i][0]) - char2value_3_17(last_cards[0])
#                 if diff > 0:
#                     mask[diff - 1] = 1
#     return mask.astype(bool)


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
        minor_cards[find_minor_in_three_one(cards) - 3] = -1
    if category_idx == Category.THREE_TWO.value:
        length = 1
        minor_cards[find_minor_in_three_two(cards) - 3] = -1
    if category_idx == Category.THREE_ONE_LINE.value:
        length = int(cards.size / 4)
        minor_cards[find_minor_in_three_one_line(cards) - 3] = -1
    if category_idx == Category.THREE_TWO_LINE.value:
        length = int(cards.size / 5)
        minor_cards[find_minor_in_three_two_line(cards) - 3] = -1
    if category_idx == Category.FOUR_TWO.value:
        length = 2
        minor_cards[find_minor_in_four_two(cards) - 3] = -1
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
    curr_cards = to_char(env.get_curr_handcards())
    curr_val, curr_round = env.get_cards_value(card.Card.char2color(curr_cards))
    if mask is None:
        mask = get_mask(curr_cards, action_space, to_char(env.get_last_outcards()))
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


class MinorCardNetwork:
    def __init__(self, s_dim, trainer, scope):
        with tf.variable_scope(scope):
            # need a slightly different state per timestep
            # assume the first dimension is [batch * timestep]
            self.batch_size = tf.placeholder(tf.int32, None, name='batch_size')
            with tf.name_scope("minor/input_state"):
                self.input_state = tf.placeholder(tf.float32, [None, s_dim], name="input")
            with tf.name_scope("minor/training"):
                self.training = tf.placeholder(tf.bool, None, name="mode")
            with tf.name_scope("minor/input_single"):
                self.input_single = tf.placeholder(tf.float32, [None, 15], name="input_single")
            with tf.name_scope("minor/input_pair"):
                self.input_pair = tf.placeholder(tf.float32, [None, 13], name="input_pair")
            with tf.name_scope("minor/input_triple"):
                self.input_triple = tf.placeholder(tf.float32, [None, 13], name="input_triple")
            with tf.name_scope("minor/input_quadric"):
                self.input_quadric = tf.placeholder(tf.float32, [None, 13], name="input_quadric")

            # TODO: test if embedding would help
            with tf.name_scope("input_state_embedding"):
                self.embeddings = slim.fully_connected(
                    inputs=self.input_state,
                    num_outputs=512,
                    activation_fn=tf.nn.elu,
                    weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("minor/reshaping_for_conv"):
                self.input_state_conv = tf.reshape(self.embeddings, [-1, 1, 512, 1])
                self.input_single_conv = tf.reshape(self.input_single, [-1, 1, 15, 1])
                self.input_pair_conv = tf.reshape(self.input_pair, [-1, 1, 13, 1])
                self.input_triple_conv = tf.reshape(self.input_triple, [-1, 1, 13, 1])
                self.input_quadric_conv = tf.reshape(self.input_quadric, [-1, 1, 13, 1])

            # convolution for legacy state
            with tf.name_scope("minor/conv_legacy_state"):
                self.state_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_state_conv,
                                                         num_outputs=16,
                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1a = tf.layers.batch_normalization(self.state_conv1a_branch1a,
                                                                         training=self.training)
                self.state_nonlinear1a_branch1a = tf.nn.relu(self.state_bn1a_branch1a)

                self.state_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.state_nonlinear1a_branch1a,
                                                         num_outputs=16,
                                                         kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1b = tf.layers.batch_normalization(self.state_conv1a_branch1b,
                                                                         training=self.training)
                self.state_nonlinear1a_branch1b = tf.nn.relu(self.state_bn1a_branch1b)

                self.state_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.state_nonlinear1a_branch1b,
                                                         num_outputs=64,
                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1c = tf.layers.batch_normalization(self.state_conv1a_branch1c,
                                                                         training=self.training)

                ######

                self.state_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_state_conv,
                                                        num_outputs=64,
                                                        kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch2 = tf.layers.batch_normalization(self.state_conv1a_branch2,
                                                                        training=self.training)

                self.state1a = self.state_bn1a_branch1c + self.state_bn1a_branch2
                self.state_output = slim.flatten(tf.nn.relu(self.state1a))

            # convolution for single
            with tf.name_scope("minor/conv_single"):
                self.single_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_single_conv,
                                                          num_outputs=16,
                                                          kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1a = tf.layers.batch_normalization(self.single_conv1a_branch1a,
                                                                          training=self.training)
                self.single_nonlinear1a_branch1a = tf.nn.relu(self.single_bn1a_branch1a)

                self.single_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1a,
                                                          num_outputs=16,
                                                          kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1b = tf.layers.batch_normalization(self.single_conv1a_branch1b,
                                                                          training=self.training)
                self.single_nonlinear1a_branch1b = tf.nn.relu(self.single_bn1a_branch1b)

                self.single_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1b,
                                                          num_outputs=64,
                                                          kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1c = tf.layers.batch_normalization(self.single_conv1a_branch1c,
                                                                          training=self.training)

                ######

                self.single_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_single_conv,
                                                         num_outputs=64,
                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch2 = tf.layers.batch_normalization(self.single_conv1a_branch2,
                                                                         training=self.training)

                self.single1a = self.single_bn1a_branch1c + self.single_bn1a_branch2
                self.single_output = slim.flatten(tf.nn.relu(self.single1a))

            # convolution for pair
            with tf.name_scope("minor/conv_pair"):
                self.pair_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=16,
                                                        kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1a = tf.layers.batch_normalization(self.pair_conv1a_branch1a,
                                                                        training=self.training)
                self.pair_nonlinear1a_branch1a = tf.nn.relu(self.pair_bn1a_branch1a)

                self.pair_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1a,
                                                        num_outputs=16,
                                                        kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1b = tf.layers.batch_normalization(self.pair_conv1a_branch1b,
                                                                        training=self.training)
                self.pair_nonlinear1a_branch1b = tf.nn.relu(self.pair_bn1a_branch1b)

                self.pair_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1b,
                                                        num_outputs=64,
                                                        kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1c = tf.layers.batch_normalization(self.pair_conv1a_branch1c,
                                                                        training=self.training)

                ######

                self.pair_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=64,
                                                       kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch2 = tf.layers.batch_normalization(self.pair_conv1a_branch2, training=self.training)

                self.pair1a = self.pair_bn1a_branch1c + self.pair_bn1a_branch2
                self.pair_output = slim.flatten(tf.nn.relu(self.pair1a))

            # convolution for triple
            with tf.name_scope("minor/conv_triple"):
                self.triple_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv,
                                                          num_outputs=16,
                                                          kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1a = tf.layers.batch_normalization(self.triple_conv1a_branch1a,
                                                                          training=self.training)
                self.triple_nonlinear1a_branch1a = tf.nn.relu(self.triple_bn1a_branch1a)

                self.triple_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1a,
                                                          num_outputs=16,
                                                          kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1b = tf.layers.batch_normalization(self.triple_conv1a_branch1b,
                                                                          training=self.training)
                self.triple_nonlinear1a_branch1b = tf.nn.relu(self.triple_bn1a_branch1b)

                self.triple_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1b,
                                                          num_outputs=64,
                                                          kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1c = tf.layers.batch_normalization(self.triple_conv1a_branch1c,
                                                                          training=self.training)

                ######

                self.triple_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv,
                                                         num_outputs=64,
                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch2 = tf.layers.batch_normalization(self.triple_conv1a_branch2,
                                                                         training=self.training)

                self.triple1a = self.triple_bn1a_branch1c + self.triple_bn1a_branch2
                self.triple_output = slim.flatten(tf.nn.relu(self.triple1a))

            # convolution for quadric
            with tf.name_scope("minor/conv_quadric"):
                self.quadric_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv,
                                                           num_outputs=16,
                                                           kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1a = tf.layers.batch_normalization(self.quadric_conv1a_branch1a,
                                                                           training=self.training)
                self.quadric_nonlinear1a_branch1a = tf.nn.relu(self.quadric_bn1a_branch1a)

                self.quadric_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1a,
                                                           num_outputs=16,
                                                           kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1b = tf.layers.batch_normalization(self.quadric_conv1a_branch1b,
                                                                           training=self.training)
                self.quadric_nonlinear1a_branch1b = tf.nn.relu(self.quadric_bn1a_branch1b)

                self.quadric_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1b,
                                                           num_outputs=64,
                                                           kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1c = tf.layers.batch_normalization(self.quadric_conv1a_branch1c,
                                                                           training=self.training)

                ######

                self.quadric_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv,
                                                          num_outputs=64,
                                                          kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch2 = tf.layers.batch_normalization(self.quadric_conv1a_branch2,
                                                                          training=self.training)

                self.quadric1a = self.quadric_bn1a_branch1c + self.quadric_bn1a_branch2
                self.quadric_output = slim.flatten(tf.nn.relu(self.quadric1a))

            # 3 + 1 convolution
            with tf.name_scope("minor/conv_3plus1"):
                tiled_triple = tf.tile(tf.expand_dims(self.input_triple, 1), [1, 15, 1])
                tiled_single = tf.tile(tf.expand_dims(self.input_single, 2), [1, 1, 13])
                self.input_triple_single_conv = tf.to_float(
                    tf.expand_dims(tf.bitwise.bitwise_and(tf.to_int32(tiled_triple),
                                                          tf.to_int32(tiled_single)), -1))

                self.triple_single_conv1a_branch1a = slim.conv2d(activation_fn=None,
                                                                 inputs=self.input_triple_single_conv, num_outputs=16,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1a = tf.layers.batch_normalization(self.triple_single_conv1a_branch1a,
                                                                                 training=self.training)
                self.triple_single_nonlinear1a_branch1a = tf.nn.relu(self.triple_single_bn1a_branch1a)

                self.triple_single_conv1a_branch1b = slim.conv2d(activation_fn=None,
                                                                 inputs=self.triple_single_nonlinear1a_branch1a,
                                                                 num_outputs=16,
                                                                 kernel_size=[3, 3], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1b = tf.layers.batch_normalization(self.triple_single_conv1a_branch1b,
                                                                                 training=self.training)
                self.triple_single_nonlinear1a_branch1b = tf.nn.relu(self.triple_single_bn1a_branch1b)

                self.triple_single_conv1a_branch1c = slim.conv2d(activation_fn=None,
                                                                 inputs=self.triple_single_nonlinear1a_branch1b,
                                                                 num_outputs=64,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1c = tf.layers.batch_normalization(self.triple_single_conv1a_branch1c,
                                                                                 training=self.training)

                ######

                self.triple_single_conv1a_branch2 = slim.conv2d(activation_fn=None,
                                                                inputs=self.input_triple_single_conv, num_outputs=64,
                                                                kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch2 = tf.layers.batch_normalization(self.triple_single_conv1a_branch2,
                                                                                training=self.training)

                self.triple_single1a = self.triple_single_bn1a_branch1c + self.triple_single_bn1a_branch2
                self.triple_single_output = slim.flatten(tf.nn.relu(self.triple_single1a))

            # 3 + 2 convolution
            with tf.name_scope("minor/conv_3plus2"):
                tiled_triple = tf.tile(tf.expand_dims(self.input_triple, 1), [1, 13, 1])
                tiled_double = tf.tile(tf.expand_dims(self.input_pair, 2), [1, 1, 13])
                self.input_triple_double_conv = tf.to_float(
                    tf.expand_dims(tf.bitwise.bitwise_and(tf.to_int32(tiled_triple),
                                                          tf.to_int32(tiled_double)), -1))

                self.triple_double_conv1a_branch1a = slim.conv2d(activation_fn=None,
                                                                 inputs=self.input_triple_double_conv, num_outputs=16,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1a = tf.layers.batch_normalization(self.triple_double_conv1a_branch1a,
                                                                                 training=self.training)
                self.triple_double_nonlinear1a_branch1a = tf.nn.relu(self.triple_double_bn1a_branch1a)

                self.triple_double_conv1a_branch1b = slim.conv2d(activation_fn=None,
                                                                 inputs=self.triple_double_nonlinear1a_branch1a,
                                                                 num_outputs=16,
                                                                 kernel_size=[3, 3], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1b = tf.layers.batch_normalization(self.triple_double_conv1a_branch1b,
                                                                                 training=self.training)
                self.triple_double_nonlinear1a_branch1b = tf.nn.relu(self.triple_double_bn1a_branch1b)

                self.triple_double_conv1a_branch1c = slim.conv2d(activation_fn=None,
                                                                 inputs=self.triple_double_nonlinear1a_branch1b,
                                                                 num_outputs=64,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1c = tf.layers.batch_normalization(self.triple_double_conv1a_branch1c,
                                                                                 training=self.training)

                ######

                self.triple_double_conv1a_branch2 = slim.conv2d(activation_fn=None,
                                                                inputs=self.input_triple_double_conv, num_outputs=64,
                                                                kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch2 = tf.layers.batch_normalization(self.triple_double_conv1a_branch2,
                                                                                training=self.training)

                self.triple_double1a = self.triple_double_bn1a_branch1c + self.triple_double_bn1a_branch2
                self.triple_double_output = slim.flatten(tf.nn.relu(self.triple_double1a))

            #################################################

            # concatenated to [(batch * t) * flattened_dimenstion] and reshape to [batch * t * flattened_dimenstion]
            with tf.name_scope("minor/concatenated"):
                self.fc_flattened = tf.concat([self.single_output, self.pair_output, self.triple_output,
                                               self.quadric_output, self.triple_single_output,
                                               self.triple_double_output, self.state_output], 1)
                self.fc_flattened = tf.reshape(self.fc_flattened, tf.stack([self.batch_size, -1, 15]))

            with tf.name_scope("minor/minor_cards"):
                # dynamic time step
                # b * t * 15
                s = self.fc_flattened
                step_cnt = tf.shape(s)[1:2]

                # a lstm network
                self.lstm = rnn.BasicLSTMCell(num_units=256, state_is_tuple=True)
                c_input = tf.placeholder(tf.float32, [None, self.lstm.state_size.c])
                h_input = tf.placeholder(tf.float32, [None, self.lstm.state_size.h])

                self.lstm_state_input = rnn.LSTMStateTuple(c_input, h_input)
                lstm_input = slim.fully_connected(inputs=s, num_outputs=64,
                                                  activation_fn=tf.nn.relu)
                self.lstm_output, self.lstm_state_output = tf.nn.dynamic_rnn(self.lstm, lstm_input,
                                                                             initial_state=self.lstm_state_input,
                                                                             sequence_length=step_cnt)
                # size: b * t * 15
                self.policy_pred = slim.fully_connected(inputs=self.lstm_output, num_outputs=15,
                                                        activation_fn=tf.nn.softmax)
                # action size: b * t
                self.action = tf.placeholder(shape=[None], dtype=tf.int32)
                self.action = tf.reshape(self.action, tf.stack([self.batch_size, -1]))
                self.action_onehot = tf.one_hot(self.action, 15, dtype=tf.float32)

                # advantage size: b * 1
                self.advantages = tf.placeholder(shape=[None, 1], dtype=tf.float32)

                # b * t prob
                self.pi_stoch = tf.reduce_sum(self.policy_pred * self.action_onehot, [2])

                # Loss functions
                # self.value_loss = tf.reduce_sum(tf.square(self.target_val - tf.reshape(self.val_pred, [-1])))
                # self.action_entropy = -tf.reduce_sum(self.policy_pred * tf.log(self.policy_pred))

                # -log(P(A) * P(B|A)...) = -log(P(A)) - log(P(B|A)) - ...
                # broadcasting with b * t and b * 1
                self.policy_loss = -tf.reduce_sum(
                    tf.log(tf.clip_by_value(self.pi_stoch, 1e-6, 1 - 1e-6)) * self.advantages)
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.policy_loss, local_vars)
                self.apply_grads = trainer.apply_gradients(zip(self.gradients, local_vars))


##################################################### UTILITIES ########################################################
class CardNetwork:
    def __init__(self, s_dim, trainer, scope, a_dim=9085):
        with tf.variable_scope(scope):
            # card_cnt = 57
            # self.temp = tf.placeholder(tf.float32, None, name="boltz")
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
                self.state_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_state_conv,
                                                         num_outputs=16,
                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1a = tf.layers.batch_normalization(self.state_conv1a_branch1a,
                                                                         training=self.training)
                self.state_nonlinear1a_branch1a = tf.nn.relu(self.state_bn1a_branch1a)

                self.state_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.state_nonlinear1a_branch1a,
                                                         num_outputs=16,
                                                         kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1b = tf.layers.batch_normalization(self.state_conv1a_branch1b,
                                                                         training=self.training)
                self.state_nonlinear1a_branch1b = tf.nn.relu(self.state_bn1a_branch1b)

                self.state_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.state_nonlinear1a_branch1b,
                                                         num_outputs=64,
                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1c = tf.layers.batch_normalization(self.state_conv1a_branch1c,
                                                                         training=self.training)

                ######

                self.state_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_state_conv,
                                                        num_outputs=64,
                                                        kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch2 = tf.layers.batch_normalization(self.state_conv1a_branch2,
                                                                        training=self.training)

                self.state1a = self.state_bn1a_branch1c + self.state_bn1a_branch2
                self.state_output = slim.flatten(tf.nn.relu(self.state1a))

            # convolution for single
            with tf.name_scope("conv_single"):
                self.single_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_single_conv,
                                                          num_outputs=16,
                                                          kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1a = tf.layers.batch_normalization(self.single_conv1a_branch1a,
                                                                          training=self.training)
                self.single_nonlinear1a_branch1a = tf.nn.relu(self.single_bn1a_branch1a)

                self.single_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1a,
                                                          num_outputs=16,
                                                          kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1b = tf.layers.batch_normalization(self.single_conv1a_branch1b,
                                                                          training=self.training)
                self.single_nonlinear1a_branch1b = tf.nn.relu(self.single_bn1a_branch1b)

                self.single_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1b,
                                                          num_outputs=64,
                                                          kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1c = tf.layers.batch_normalization(self.single_conv1a_branch1c,
                                                                          training=self.training)

                ######

                self.single_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_single_conv,
                                                         num_outputs=64,
                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch2 = tf.layers.batch_normalization(self.single_conv1a_branch2,
                                                                         training=self.training)

                self.single1a = self.single_bn1a_branch1c + self.single_bn1a_branch2
                self.single_output = slim.flatten(tf.nn.relu(self.single1a))

            # convolution for pair
            with tf.name_scope("conv_pair"):
                self.pair_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=16,
                                                        kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1a = tf.layers.batch_normalization(self.pair_conv1a_branch1a,
                                                                        training=self.training)
                self.pair_nonlinear1a_branch1a = tf.nn.relu(self.pair_bn1a_branch1a)

                self.pair_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1a,
                                                        num_outputs=16,
                                                        kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1b = tf.layers.batch_normalization(self.pair_conv1a_branch1b,
                                                                        training=self.training)
                self.pair_nonlinear1a_branch1b = tf.nn.relu(self.pair_bn1a_branch1b)

                self.pair_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1b,
                                                        num_outputs=64,
                                                        kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1c = tf.layers.batch_normalization(self.pair_conv1a_branch1c,
                                                                        training=self.training)

                ######

                self.pair_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=64,
                                                       kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch2 = tf.layers.batch_normalization(self.pair_conv1a_branch2, training=self.training)

                self.pair1a = self.pair_bn1a_branch1c + self.pair_bn1a_branch2
                self.pair_output = slim.flatten(tf.nn.relu(self.pair1a))

            # convolution for triple
            with tf.name_scope("conv_triple"):
                self.triple_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv,
                                                          num_outputs=16,
                                                          kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1a = tf.layers.batch_normalization(self.triple_conv1a_branch1a,
                                                                          training=self.training)
                self.triple_nonlinear1a_branch1a = tf.nn.relu(self.triple_bn1a_branch1a)

                self.triple_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1a,
                                                          num_outputs=16,
                                                          kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1b = tf.layers.batch_normalization(self.triple_conv1a_branch1b,
                                                                          training=self.training)
                self.triple_nonlinear1a_branch1b = tf.nn.relu(self.triple_bn1a_branch1b)

                self.triple_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1b,
                                                          num_outputs=64,
                                                          kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1c = tf.layers.batch_normalization(self.triple_conv1a_branch1c,
                                                                          training=self.training)

                ######

                self.triple_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv,
                                                         num_outputs=64,
                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch2 = tf.layers.batch_normalization(self.triple_conv1a_branch2,
                                                                         training=self.training)

                self.triple1a = self.triple_bn1a_branch1c + self.triple_bn1a_branch2
                self.triple_output = slim.flatten(tf.nn.relu(self.triple1a))

            # convolution for quadric
            with tf.name_scope("conv_quadric"):
                self.quadric_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv,
                                                           num_outputs=16,
                                                           kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1a = tf.layers.batch_normalization(self.quadric_conv1a_branch1a,
                                                                           training=self.training)
                self.quadric_nonlinear1a_branch1a = tf.nn.relu(self.quadric_bn1a_branch1a)

                self.quadric_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1a,
                                                           num_outputs=16,
                                                           kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1b = tf.layers.batch_normalization(self.quadric_conv1a_branch1b,
                                                                           training=self.training)
                self.quadric_nonlinear1a_branch1b = tf.nn.relu(self.quadric_bn1a_branch1b)

                self.quadric_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1b,
                                                           num_outputs=64,
                                                           kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1c = tf.layers.batch_normalization(self.quadric_conv1a_branch1c,
                                                                           training=self.training)

                ######

                self.quadric_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv,
                                                          num_outputs=64,
                                                          kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch2 = tf.layers.batch_normalization(self.quadric_conv1a_branch2,
                                                                          training=self.training)

                self.quadric1a = self.quadric_bn1a_branch1c + self.quadric_bn1a_branch2
                self.quadric_output = slim.flatten(tf.nn.relu(self.quadric1a))

            # 3 + 1 convolution
            with tf.name_scope("conv_3plus1"):
                tiled_triple = tf.tile(tf.expand_dims(self.input_triple, 1), [1, 15, 1])
                tiled_single = tf.tile(tf.expand_dims(self.input_single, 2), [1, 1, 13])
                self.input_triple_single_conv = tf.to_float(
                    tf.expand_dims(tf.bitwise.bitwise_and(tf.to_int32(tiled_triple),
                                                          tf.to_int32(tiled_single)), -1))

                self.triple_single_conv1a_branch1a = slim.conv2d(activation_fn=None,
                                                                 inputs=self.input_triple_single_conv, num_outputs=16,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1a = tf.layers.batch_normalization(self.triple_single_conv1a_branch1a,
                                                                                 training=self.training)
                self.triple_single_nonlinear1a_branch1a = tf.nn.relu(self.triple_single_bn1a_branch1a)

                self.triple_single_conv1a_branch1b = slim.conv2d(activation_fn=None,
                                                                 inputs=self.triple_single_nonlinear1a_branch1a,
                                                                 num_outputs=16,
                                                                 kernel_size=[3, 3], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1b = tf.layers.batch_normalization(self.triple_single_conv1a_branch1b,
                                                                                 training=self.training)
                self.triple_single_nonlinear1a_branch1b = tf.nn.relu(self.triple_single_bn1a_branch1b)

                self.triple_single_conv1a_branch1c = slim.conv2d(activation_fn=None,
                                                                 inputs=self.triple_single_nonlinear1a_branch1b,
                                                                 num_outputs=64,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1c = tf.layers.batch_normalization(self.triple_single_conv1a_branch1c,
                                                                                 training=self.training)

                ######

                self.triple_single_conv1a_branch2 = slim.conv2d(activation_fn=None,
                                                                inputs=self.input_triple_single_conv, num_outputs=64,
                                                                kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch2 = tf.layers.batch_normalization(self.triple_single_conv1a_branch2,
                                                                                training=self.training)

                self.triple_single1a = self.triple_single_bn1a_branch1c + self.triple_single_bn1a_branch2
                self.triple_single_output = slim.flatten(tf.nn.relu(self.triple_single1a))

            # 3 + 2 convolution
            with tf.name_scope("conv_3plus2"):
                tiled_triple = tf.tile(tf.expand_dims(self.input_triple, 1), [1, 13, 1])
                tiled_double = tf.tile(tf.expand_dims(self.input_pair, 2), [1, 1, 13])
                self.input_triple_double_conv = tf.to_float(
                    tf.expand_dims(tf.bitwise.bitwise_and(tf.to_int32(tiled_triple),
                                                          tf.to_int32(tiled_double)), -1))

                self.triple_double_conv1a_branch1a = slim.conv2d(activation_fn=None,
                                                                 inputs=self.input_triple_double_conv, num_outputs=16,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1a = tf.layers.batch_normalization(self.triple_double_conv1a_branch1a,
                                                                                 training=self.training)
                self.triple_double_nonlinear1a_branch1a = tf.nn.relu(self.triple_double_bn1a_branch1a)

                self.triple_double_conv1a_branch1b = slim.conv2d(activation_fn=None,
                                                                 inputs=self.triple_double_nonlinear1a_branch1a,
                                                                 num_outputs=16,
                                                                 kernel_size=[3, 3], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1b = tf.layers.batch_normalization(self.triple_double_conv1a_branch1b,
                                                                                 training=self.training)
                self.triple_double_nonlinear1a_branch1b = tf.nn.relu(self.triple_double_bn1a_branch1b)

                self.triple_double_conv1a_branch1c = slim.conv2d(activation_fn=None,
                                                                 inputs=self.triple_double_nonlinear1a_branch1b,
                                                                 num_outputs=64,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1c = tf.layers.batch_normalization(self.triple_double_conv1a_branch1c,
                                                                                 training=self.training)

                ######

                self.triple_double_conv1a_branch2 = slim.conv2d(activation_fn=None,
                                                                inputs=self.input_triple_double_conv, num_outputs=64,
                                                                kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch2 = tf.layers.batch_normalization(self.triple_double_conv1a_branch2,
                                                                                training=self.training)

                self.triple_double1a = self.triple_double_bn1a_branch1c + self.triple_double_bn1a_branch2
                self.triple_double_output = slim.flatten(tf.nn.relu(self.triple_double1a))

            #################################################

            with tf.name_scope("concated"):
                self.fc_flattened = tf.concat([self.single_output, self.pair_output, self.triple_output,
                                               self.quadric_output, self.triple_single_output,
                                               self.triple_double_output, self.state_output], 1)

            # passive decision making  0: pass, 1: bomb, 2: king, 3: normal
            with tf.name_scope("passive_decision_making"):
                self.fc_decision_passive = slim.fully_connected(self.fc_flattened, 128, activation_fn=tf.nn.relu)
                self.fc_decision_passive = slim.fully_connected(self.fc_decision_passive, 64, activation_fn=tf.nn.relu)
                self.fc_decision_passive = slim.fully_connected(self.fc_decision_passive, 32, activation_fn=tf.nn.relu)
                self.fc_decision_passive_output = slim.fully_connected(self.fc_decision_passive, 4, tf.nn.softmax)

            # passive response
            with tf.name_scope("passive_response"):
                self.fc_response_passive = slim.fully_connected(self.fc_flattened, 128, activation_fn=tf.nn.relu)
                self.fc_response_passive = slim.fully_connected(self.fc_response_passive, 64, activation_fn=tf.nn.relu)
                self.fc_response_passive = slim.fully_connected(self.fc_response_passive, 32, activation_fn=tf.nn.relu)
                self.fc_response_passive_output = slim.fully_connected(self.fc_response_passive, 14, tf.nn.softmax)

            # passive bomb response
            with tf.name_scope("passive_bomb_reponse"):
                self.fc_bomb_passive = slim.fully_connected(self.fc_flattened, 128, activation_fn=tf.nn.relu)
                self.fc_bomb_passive = slim.fully_connected(self.fc_bomb_passive, 64, activation_fn=tf.nn.relu)
                self.fc_bomb_passive = slim.fully_connected(self.fc_bomb_passive, 32, activation_fn=tf.nn.relu)
                self.fc_bomb_passive_output = slim.fully_connected(self.fc_bomb_passive, 13, tf.nn.softmax)

            # active decision making  mapped to [action space category - 1]
            with tf.name_scope("active_decision_making"):
                self.fc_decision_active = slim.fully_connected(self.fc_flattened, 128, activation_fn=tf.nn.relu)
                self.fc_decision_active = slim.fully_connected(self.fc_decision_active, 64, activation_fn=tf.nn.relu)
                self.fc_decision_active = slim.fully_connected(self.fc_decision_active, 32, activation_fn=tf.nn.relu)
                self.fc_decision_active_output = slim.fully_connected(self.fc_decision_active, 13, tf.nn.softmax)

            # active response
            with tf.name_scope("active_response"):
                self.fc_response_active = slim.fully_connected(self.fc_flattened, 128, activation_fn=tf.nn.relu)
                self.fc_response_active = slim.fully_connected(self.fc_response_active, 64, activation_fn=tf.nn.relu)
                self.fc_response_active = slim.fully_connected(self.fc_response_active, 32, activation_fn=tf.nn.relu)
                self.fc_response_active_output = slim.fully_connected(self.fc_response_active, 15, tf.nn.softmax)

            # card length output
            with tf.name_scope("fc_sequence_length_output"):
                self.fc_seq_length = slim.fully_connected(self.fc_flattened, 128, activation_fn=tf.nn.relu)
                self.fc_seq_length = slim.fully_connected(self.fc_seq_length, 64, activation_fn=tf.nn.relu)
                self.fc_seq_length = slim.fully_connected(self.fc_seq_length, 32, activation_fn=tf.nn.relu)
                self.fc_seq_length_output = slim.fully_connected(self.fc_seq_length, 12, tf.nn.softmax)

            # value output
            with tf.name_scope("fc_value_output"):
                self.fc_value = slim.fully_connected(self.fc_flattened, 128, activation_fn=tf.nn.relu)
                self.fc_value = slim.fully_connected(self.fc_value, 64, activation_fn=tf.nn.relu)
                self.fc_value = slim.fully_connected(self.fc_value, 32, activation_fn=tf.nn.relu)
                self.fc_value = slim.fully_connected(self.fc_value, 8, activation_fn=tf.nn.relu)
                self.fc_value_output = slim.fully_connected(self.fc_value, 1, tf.nn.elu)

            # input values
            with tf.name_scope("input_values"):
                self.val_truth = tf.placeholder(tf.float32, [None], name='val_input')

            # advantage functions
            with tf.name_scope("input_advantages"):
                self.advantages = tf.placeholder(tf.float32, [None], name='advantage_input')

            # active or passive
            with tf.name_scope("input_is_active"):
                self.is_active = tf.placeholder(tf.bool, [None], name='active')

            with tf.name_scope("has_seq_length"):
                self.has_seq_length = tf.placeholder(tf.bool, [None], name='has_seq_length')

            # card length - max len: 12 (3-A)
            with tf.name_scope("sequence_length"):
                self.length_input = tf.placeholder(tf.int32, [None], name='sequence_length')
                self.length_target = tf.one_hot(self.length_input, 12)
                self.length_sample = tf.reduce_sum(tf.multiply(self.length_target, self.fc_seq_length_output), 1)
                self.length_loss = -tf.log(tf.clip_by_value(self.length_sample, 1e-8, 1-1e-8)) * self.advantages

            # passive mode
            with tf.name_scope("passive_mode_loss"):
                self.is_passive_bomb = tf.placeholder(tf.bool, [None], name='passive_bomb')
                self.is_passive_king = tf.placeholder(tf.bool, [None], name='passive_is_king')

                self.passive_decision_input = tf.placeholder(tf.int32, [None], name='passive_decision_in')
                self.passive_decision_target = tf.one_hot(self.passive_decision_input, 4)
                self.passive_decision_sample = tf.reduce_sum(tf.multiply(self.passive_decision_target,
                                                                         self.fc_decision_passive_output), 1)
                self.passive_decision_loss = -tf.log(tf.clip_by_value(self.passive_decision_sample, 1e-8, 1-1e-8)) * self.advantages

                self.passive_response_input = tf.placeholder(tf.int32, [None], name='passive_response_in')
                self.passive_response_target = tf.one_hot(self.passive_response_input, 14)
                self.passive_response_sample = tf.reduce_sum(tf.multiply(self.passive_response_target,
                                                                         self.fc_response_passive_output), 1)
                self.passive_response_loss = -tf.log(tf.clip_by_value(self.passive_response_sample, 1e-8, 1-1e-8)) * self.advantages

                self.passive_bomb_input = tf.placeholder(tf.int32, [None], name='passive_bomb_in')
                self.passive_bomb_target = tf.one_hot(self.passive_bomb_input, 13)
                self.passive_bomb_sample = tf.reduce_sum(tf.multiply(self.passive_bomb_target,
                                                                     self.fc_bomb_passive_output), 1)
                self.passive_bomb_loss = -tf.log(tf.clip_by_value(self.passive_bomb_sample, 1e-8, 1-1e-8)) * self.advantages

            # active mode
            with tf.name_scope("active_mode_loss"):
                self.active_decision_input = tf.placeholder(tf.int32, [None], name='active_decision_in')
                self.active_decision_target = tf.one_hot(self.active_decision_input, 13)
                self.active_decision_sample = tf.reduce_sum(tf.multiply(self.active_decision_target,
                                                                        self.fc_decision_active_output), 1)
                self.active_decision_loss = -tf.log(tf.clip_by_value(self.active_decision_sample, 1e-8, 1-1e-8)) * self.advantages

                self.active_response_input = tf.placeholder(tf.int32, [None], name='active_response_in')
                self.active_response_target = tf.one_hot(self.active_response_input, 15)
                self.active_response_sample = tf.reduce_sum(tf.multiply(self.active_response_target,
                                                                        self.fc_response_active_output), 1)
                self.active_response_loss = -tf.log(tf.clip_by_value(self.active_response_sample, 1e-8, 1-1e-8)) * self.advantages

            with tf.name_scope("passive_loss"):
                self.passive_loss = self.passive_decision_loss + (1 - tf.to_float(self.is_passive_king)) * (
                    (1 - tf.to_float(self.is_passive_bomb)) * self.passive_response_loss + \
                    tf.to_float(self.is_passive_bomb) * self.passive_bomb_loss)

            with tf.name_scope("active_loss"):
                self.active_loss = self.active_decision_loss + self.active_response_loss

            with tf.name_scope("value_loss"):
                self.val_loss = 0.2 * tf.reduce_sum(tf.square(self.fc_value_output - self.val_truth))

            # policy loss (active + passive + length) + value loss
            with tf.name_scope("total_loss"):
                self.loss = tf.reduce_sum(tf.to_float(self.is_active) * self.active_loss + (1 - tf.to_float(
                    self.is_active)) * self.passive_loss + tf.to_float(self.has_seq_length) * self.length_loss) + self.val_loss

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
        self.main_network = CardNetwork(54 * 6, trainer, self.name, 8310)
        self.minor_network = MinorCardNetwork(54 * 6, trainer, self.name + '/minor')
        self.action_space_single = action_space[1:16]
        self.action_space_pair = action_space[16:29]
        self.action_space_triple = action_space[29:42]
        self.action_space_quadric = action_space[42:55]

    def train_batch_packed(self, buffer, sess, gamma, val_last):
        rewards = np.array([buffer[i][1] for i in range(len(buffer))])
        values = np.array([buffer[i][2] for i in range(len(buffer))])

        rewards_plus = np.append(rewards, val_last)
        val_truth = discounted_return(rewards_plus, gamma)[:-1]

        val_pred_plus = np.append(values, val_last)
        td0 = rewards + gamma * val_pred_plus[1:] - val_pred_plus[:-1]
        advantages = discounted_return(td0, gamma)

        records = []

        for i in range(len(buffer)):
            buffer[i][18] = advantages[i]
            buffer[i][19] = [val_truth[i]]
            buff = buffer[i]
            records.append(self.train_batch(buff, sess))

        return records

    def train_batch(self, buffer, sess):

        training, input_state, input_single, input_pair, input_triple, \
            input_quadric, is_active, has_seq_length, seq_length_input, \
            is_passive_bomb, is_passive_king, passive_decision_input, \
            passive_response_input, passive_bomb_input, active_decision_input, \
            active_response_input, advantages, val_truth, has_minor_cards = [buffer[i] for i in range(2, 21)]

        # main network training
        decision_passive_output, response_passive_output, bomb_passive_output, \
        decision_active_output, response_active_output, main_loss, main_val_loss, \
        active_decision_loss, active_response_loss, passive_decision_loss, passive_response_loss, passive_bomb_loss, main_grads, _ \
            = sess.run([self.main_network.fc_decision_passive_output,
                        self.main_network.fc_response_passive_output, self.main_network.fc_bomb_passive_output,
                        self.main_network.fc_decision_active_output, self.main_network.fc_response_active_output,
                        self.main_network.loss, self.main_network.val_loss,
                        self.main_network.active_decision_loss, self.main_network.active_response_loss, self.main_network.passive_decision_loss,
                        self.main_network.passive_response_loss, self.main_network.passive_bomb_loss, self.main_network.gradients,
                        self.main_network.optimize],
                       feed_dict={
                           self.main_network.training: training,
                           self.main_network.input_state: input_state,
                           self.main_network.input_single: input_single.reshape(1, -1),
                           self.main_network.input_pair: input_pair.reshape(1, -1),
                           self.main_network.input_triple: input_triple.reshape(1, -1),
                           self.main_network.input_quadric: input_quadric.reshape(1, -1),
                           self.main_network.val_truth: val_truth,
                           self.main_network.advantages: advantages,
                           self.main_network.is_active: np.array([is_active]),
                           self.main_network.has_seq_length: np.array([has_seq_length]),
                           self.main_network.length_input: np.array([seq_length_input]),
                           self.main_network.is_passive_bomb: np.array([is_passive_bomb]),
                           self.main_network.is_passive_king: np.array([is_passive_king]),
                           self.main_network.passive_decision_input: np.array([passive_decision_input]),
                           self.main_network.passive_response_input: np.array([passive_response_input]),
                           self.main_network.passive_bomb_input: np.array([passive_bomb_input]),
                           self.main_network.active_decision_input: np.array([active_decision_input]),
                           self.main_network.active_response_input: np.array([active_response_input])
                       })

        minor_policy_loss = 0
        if has_minor_cards:
            # minor network
            batch_size = input_state.shape[0]
            # get (b * t) packed input states
            input_minor_states, input_minor_single, input_minor_pair, input_minor_triple, \
                input_minor_quadric, actions = [buffer[i] for i in range(21, 26)]
            c_init = np.zeros((1, self.minor_network.lstm.state_size.c), np.float32)
            h_init = np.zeros((1, self.minor_network.lstm.state_size.h), np.float32)
            rnn_state = [c_init, h_init]
            _, minor_policy_loss = sess.run([self.minor_network.apply_grads, self.minor_network.policy_loss], feed_dict={
                self.minor_network.training: True,
                self.minor_network.input_state: input_minor_states,
                self.minor_network.input_single: input_minor_single,
                self.minor_network.input_pair: input_minor_pair,
                self.minor_network.input_triple: input_minor_triple,
                self.minor_network.input_quadric: input_minor_quadric,
                self.minor_network.lstm_state_input: rnn_state,
                self.minor_network.action: actions,
                self.minor_network.advantages: advantages,
                self.minor_network.batch_size: batch_size
            })

        episode = sess.run(self.episodes)
        return [decision_passive_output, response_passive_output, bomb_passive_output,
                decision_active_output, response_active_output, main_loss, main_val_loss,
                active_decision_loss, active_response_loss, passive_decision_loss, passive_response_loss, passive_bomb_loss,
                main_grads, minor_policy_loss]


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

        self.train_intervals = 1

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.episode_rewards = [[] for i in range(2)]
        self.episode_length = [[] for i in range(2)]
        self.episode_mean_values = [[] for i in range(2)]
        self.summary_writers = [tf.summary.FileWriter("train_agent%d" % i) for i in range(2)]

        self.agents = [CardAgent('agent%d' % i, self.trainer) for i in range(2)]

        self.global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        self.increment = self.global_episodes.assign_add(1)

    def train_batch(self, buffer, sess, gamma, val_last, idx):
        buffer = np.array(buffer)
        return self.agents[idx].train_batch(buffer, sess, gamma, val_last)

    def train_batch_packed(self, buffer, sess, gamma, val_last, idx):
        return self.agents[idx].train_batch_packed(buffer, sess, gamma, val_last)

    def respond(self, env):
        mask = get_mask(to_char(self.env.get_curr_handcards()), self.action_space, to_char(self.env.get_last_outcards()))
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
    def run(self, sess, saver, max_episode_length):
        self.sess = sess
        with sess.as_default():
            global_episodes = sess.run(self.global_episodes)
            total_episodes = 1
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
                last_category_idx = -1

                for l in range(max_episode_length):
                    # time.sleep(1)
                    # # map 1, 3 to 0, 1
                    train_id = self.env.get_role_ID()
                    train_id = int((train_id - 1) / 2)

                    print("turn %d" % l)
                    print("training id %d" % train_id)

                    # get current hand cards and last opponent's cards if any
                    curr_cards_value = self.env.get_curr_handcards()
                    curr_cards_char = to_char(curr_cards_value)
                    last_cards_value = self.env.get_last_outcards()
                    last_cards_char = to_char(last_cards_value)

                    # these are the input mask that are needed to be passed through the network
                    input_single = get_mask(curr_cards_char, action_space_single, None)
                    input_pair = get_mask(curr_cards_char, action_space_pair, None)
                    input_triple = get_mask(curr_cards_char, action_space_triple, None)
                    input_quadric = get_mask(curr_cards_char, action_space_quadric, None)

                    # first, output action through network once
                    # we use split two networks for main cards and minor cards
                    decision_passive_output, response_passive_output, bomb_passive_output, \
                    decision_active_output, response_active_output, seq_length_output, val_output \
                        = self.sess.run([self.agents[train_id].main_network.fc_decision_passive_output,
                                         self.agents[train_id].main_network.fc_response_passive_output, self.agents[train_id].main_network.fc_bomb_passive_output,
                                         self.agents[train_id].main_network.fc_decision_active_output, self.agents[train_id].main_network.fc_response_active_output,
                                         self.agents[train_id].main_network.fc_seq_length_output, self.agents[train_id].main_network.fc_value_output],
                                        feed_dict={
                                            self.agents[train_id].main_network.training: False,
                                            self.agents[train_id].main_network.input_state: s,
                                            self.agents[train_id].main_network.input_single: np.reshape(input_single, [1, -1]),
                                            self.agents[train_id].main_network.input_pair: np.reshape(input_pair, [1, -1]),
                                            self.agents[train_id].main_network.input_triple: np.reshape(input_triple, [1, -1]),
                                            self.agents[train_id].main_network.input_quadric: np.reshape(input_quadric, [1, -1])
                                        })

                    # now we have output, need to do complicated logic checking to get what to respond
                    # we will fill intention with 3-17 card value to pass to the environment
                    # also generate training batch target



                    training = True
                    input_state = s[0]
                    is_active = False
                    has_seq_length = False
                    seq_length_input = 0
                    is_passive_bomb = False
                    is_passive_king = False
                    passive_decision_input = 0
                    passive_response_input = 0
                    passive_bomb_input = 0
                    active_decision_input = 0
                    active_response_input = 0
                    has_minor_cards = False
                    minor_cards_length = 1

                    intention = None
                    input_minors_train = [0, 0, 0, 0, 0]
                    action_minors_train = 0
                    if last_cards_value.size > 0:
                        print("passive: last idx %d" % last_category_idx)
                        print("passive: last cards", end='')
                        print(last_cards_value)
                        is_bomb = False
                        if len(last_cards_value) == 4 and len(set(last_cards_value)) == 1:
                            is_bomb = True
                        decision_mask, response_mask, bomb_mask = get_mask_alter(curr_cards_char, last_cards_char,
                                                                                 is_bomb, last_category_idx)
                        decision_passive_output = decision_passive_output[0] * decision_mask
                        decision_passive = np.random.choice(4, 1, p=decision_passive_output / decision_passive_output.sum())[0]

                        # save to buffer
                        passive_decision_input = decision_passive
                        if decision_passive == 0:
                            intention = np.array([])
                        elif decision_passive == 1:
                            is_passive_bomb = True
                            bomb_passive_output = bomb_passive_output[0] * bomb_mask

                            # save to buffer
                            passive_bomb_input = np.random.choice(13, 1, p=bomb_passive_output / bomb_passive_output.sum())[0]

                            # converting 0-based index to 3-based value
                            intention = np.array([passive_bomb_input + 3] * 4)
                        elif decision_passive == 2:
                            is_passive_king = True
                            intention = np.array([16, 17])
                        elif decision_passive == 3:
                            response_passive_output = response_passive_output[0] * response_mask

                            # save to buffer
                            passive_response_input = np.random.choice(14, 1, p=response_passive_output / response_passive_output.sum())[0]
                            # there is an offset when converting from 0-based index to 1-based index
                            bigger = passive_response_input + 1

                            minor_cards_cnt = 0

                            action_minors_train = None
                            # if we have minor cards
                            if last_category_idx == Category.THREE_ONE.value or \
                                    last_category_idx == Category.THREE_TWO.value or \
                                    last_category_idx == Category.THREE_ONE_LINE.value or \
                                    last_category_idx == Category.THREE_TWO_LINE.value or \
                                    last_category_idx == Category.FOUR_TWO.value:
                                # save to buffer
                                has_minor_cards = True

                                # length out is calculated according to last output cards
                                if last_category_idx == Category.THREE_ONE.value or \
                                                last_category_idx == Category.THREE_TWO.value:
                                    minor_cards_cnt = 1
                                if last_category_idx == Category.FOUR_TWO.value:
                                    minor_cards_cnt = 2
                                if last_category_idx == Category.THREE_ONE_LINE.value:
                                    minor_cards_cnt = int(last_cards_value.size / 4)
                                if last_category_idx == Category.THREE_TWO_LINE.value:
                                    minor_cards_cnt = int(last_cards_value.size / 5)

                                # save to buffer OFFSET by one
                                minor_cards_length = minor_cards_cnt - 1

                                # feed to the minor network to get minor cards
                                c_init = np.zeros((1, self.agents[train_id].minor_network.lstm.state_size.c),
                                                  np.float32)
                                h_init = np.zeros((1, self.agents[train_id].minor_network.lstm.state_size.h),
                                                  np.float32)
                                rnn_state = [c_init, h_init]

                                # since batch size is 1 the first dimension is the time step
                                input_minors_train = [np.zeros([minor_cards_cnt, 15]), np.zeros([minor_cards_cnt, 13]),
                                                      np.zeros([minor_cards_cnt, 13]), np.zeros([minor_cards_cnt, 13]),  np.zeros([minor_cards_cnt, s.shape[1]])]
                                action_minors_train = np.zeros([minor_cards_cnt])
                                # feed step by step
                                # TODO: change state step by step
                                for j in range(minor_cards_cnt):
                                    input_minors_train[0, j, :] = input_single
                                    input_minors_train[1, j, :] = input_pair
                                    input_minors_train[2, j, :] = input_triple
                                    input_minors_train[3, j, :] = input_quadric
                                    input_minors_train[4, j, :] = s[0]
                                    policy_pred, rnn_state = self.sess.run([self.agents[train_id].minor_network.policy_pred,
                                                   self.agents[train_id].minor_network.lstm_state_output],
                                              feed_dict={
                                                  self.agents[train_id].minor_network.training: False,
                                                  self.agents[train_id].minor_network.input_state: s,
                                                  self.agents[train_id].minor_network.input_single: input_minors_train[0][j:j+1, :],
                                                  self.agents[train_id].minor_network.input_pair: input_minors_train[1][j:j+1, :],
                                                  self.agents[train_id].minor_network.input_triple: input_minors_train[2][j:j+1, :],
                                                  self.agents[train_id].minor_network.input_quadric: input_minors_train[3][j:j+1, :],
                                                  self.agents[train_id].minor_network.lstm_state_input : rnn_state,
                                                  self.agents[train_id].minor_network.batch_size : 1
                                              })
                                    p = policy_pred[0][0]
                                    a = np.random.choice(15, 1, p=p)[0]
                                    action_minors_train[j] = a

                            intention = give_cards_with_minor(bigger, action_minors_train, curr_cards_value, last_cards_value, last_category_idx, 0)
                    else:
                        is_active = True
                        decision_mask, response_mask, _ = get_mask_alter(curr_cards_char, [], False, last_category_idx)
                        # first the decision with argmax applied
                        decision_active_output = decision_active_output[0] * decision_mask

                        # save to buffer
                        active_decision_input = np.random.choice(13, 1, p=decision_active_output / decision_active_output.sum())[0]

                        decision_active = active_decision_input

                        # then convert 0-based decision_active_output to 1-based (empty eliminated) category idx
                        active_category_idx = active_decision_input + 1

                        # then the actual response to represent card value
                        response_active_output = response_active_output[0] * response_mask[decision_active]

                        # save to buffer
                        active_response_input = np.random.choice(15, 1, p=response_active_output / response_active_output.sum())[0]

                        # save to buffer
                        seq_length_input = np.random.choice(12, 1, p=seq_length_output[0] / seq_length_output[0].sum())[0]

                        # seq length only has OFFSET 1
                        seq_length_out = seq_length_input + 1
                        seq_length_out = int(seq_length_out)

                        if active_category_idx == Category.SINGLE_LINE.value or \
                                        active_category_idx == Category.DOUBLE_LINE.value or \
                                        active_category_idx == Category.TRIPLE_LINE.value or \
                                        active_category_idx == Category.THREE_ONE_LINE.value or \
                                        active_category_idx == Category.THREE_TWO_LINE.value:
                            has_seq_length = True

                        action_minors_train = None
                        # if we have minor cards
                        if active_category_idx == Category.THREE_ONE.value or \
                                        active_category_idx == Category.THREE_TWO.value or \
                                        active_category_idx == Category.THREE_ONE_LINE.value or \
                                        active_category_idx == Category.THREE_TWO_LINE.value or \
                                        active_category_idx == Category.FOUR_TWO.value:
                            # save to buffer
                            has_minor_cards = True

                            # minor cards length depend on both card type and sequence length
                            minor_cards_cnt = 0
                            if active_category_idx == Category.THREE_ONE.value or \
                                            active_category_idx == Category.THREE_TWO.value:
                                minor_cards_cnt = 1
                            if active_category_idx == Category.FOUR_TWO.value:
                                minor_cards_cnt = 2
                            if active_category_idx == Category.THREE_ONE_LINE.value:
                                minor_cards_cnt = seq_length_out
                            if active_category_idx == Category.THREE_TWO_LINE.value:
                                minor_cards_cnt = seq_length_out

                            minor_cards_length = minor_cards_cnt - 1

                            # feed to the minor network to get minor cards
                            c_init = np.zeros((1, self.agents[train_id].minor_network.lstm.state_size.c),
                                              np.float32)
                            h_init = np.zeros((1, self.agents[train_id].minor_network.lstm.state_size.h),
                                              np.float32)
                            rnn_state = [c_init, h_init]

                            # since batch size is 1 the first dimension is the time step
                            input_minors_train = [np.zeros([minor_cards_cnt, 15]), np.zeros([minor_cards_cnt, 13]),
                                                  np.zeros([minor_cards_cnt, 13]), np.zeros([minor_cards_cnt, 13]), np.zeros([minor_cards_cnt, s.shape[1]])]
                            action_minors_train = np.zeros([minor_cards_cnt])
                            # feed step by step
                            for j in range(minor_cards_cnt):
                                input_minors_train[0, j, :] = input_single
                                input_minors_train[1, j, :] = input_pair
                                input_minors_train[2, j, :] = input_triple
                                input_minors_train[3, j, :] = input_quadric
                                input_minors_train[4, j, :] = s[0]
                                polict_pred, rnn_state = self.sess.run([self.agents[train_id].minor_network.policy_pred,
                                                                        self.agents[train_id].minor_network.lstm_state_output],
                                                                       feed_dict={
                                                                           self.agents[train_id].minor_network.training: False,
                                                                           self.agents[train_id].minor_network.input_state: s,
                                                                           self.agents[train_id].minor_network.input_single:
                                                                               input_minors_train[0][j:j + 1, :],
                                                                           self.agents[train_id].minor_network.input_pair:
                                                                               input_minors_train[1][j:j + 1, :],
                                                                           self.agents[train_id].minor_network.input_triple:
                                                                               input_minors_train[2][j:j + 1, :],
                                                                           self.agents[train_id].minor_network.input_quadric:
                                                                               input_minors_train[3][j:j + 1, :],
                                                                           self.agents[train_id].minor_network.lstm_state_input: rnn_state,
                                                                           self.agents[train_id].minor_network.batch_size: 1
                                                                       })
                                p = polict_pred[0][0]
                                a = np.random.choice(15, 1, p=p)[0]
                                action_minors_train[j] = a
                        intention = give_cards_with_minor(active_response_input, action_minors_train, curr_cards_value, last_cards_value, active_category_idx, seq_length_out)

                    print(curr_cards_value)
                    print(intention)
                    # next pass through the environment
                    r, done, _ = self.env.step(cards=intention)

                    last_category_idx = self.env.get_last_outcategory_idx()
                    print("end turn")

                    # gather buffer

                    # training, input_state, input_single, input_pair, input_triple, \
                    # input_quadric, is_active, has_seq_length, seq_length_input, \
                    # is_passive_bomb, is_passive_king, passive_decision_input, \
                    # passive_response_input, passive_bomb_input, active_decision_input, \
                    # active_response_input, advantages, val_truth, has_minor_cards = [buffer[i] for i in range(2, 21)]

                    # input_minor_states, input_minor_single, input_minor_pair, input_minor_triple, \
                    # input_minor_quadric, actions = [buffer[i] for i in range(21, 26)]
                    s_prime = self.env.get_state()
                    s_prime = np.reshape(s_prime, [1, -1])

                    # leave empty for advantages and val_truth
                    episode_buffer[train_id].append([r, val_output[0], training, s, input_single, input_pair, input_triple,
                                                     input_quadric, is_active, has_seq_length, seq_length_input, is_passive_bomb,
                                                     is_passive_king, passive_decision_input, passive_response_input, passive_bomb_input,
                                                     active_decision_input, active_response_input, 0, 0, has_minor_cards, input_minors_train[4],
                                                     input_minors_train[0], input_minors_train[1], input_minors_train[2], input_minors_train[3],
                                                     action_minors_train])

                    episode_values[train_id].append(val_output[0])
                    episode_reward[train_id] += r
                    episode_steps[train_id] += 1

                    if done:
                        for i in range(2):
                            if len(episode_buffer[i]) != 0:
                                self.train_batch_packed(episode_buffer[i], sess, self.gamma, 0, i)
                        break

                    s = s_prime

                    if len(episode_buffer[train_id]) == self.train_intervals:
                        val_last = self.sess.run([self.agents[train_id].main_network.fc_value_output],
                                            feed_dict={
                                                self.agents[train_id].main_network.training: False,
                                                self.agents[train_id].main_network.input_state: s,
                                                self.agents[train_id].main_network.input_single: np.reshape(
                                                    input_single, [1, -1]),
                                                self.agents[train_id].main_network.input_pair: np.reshape(input_pair,
                                                                                                          [1, -1]),
                                                self.agents[train_id].main_network.input_triple: np.reshape(
                                                    input_triple, [1, -1]),
                                                self.agents[train_id].main_network.input_quadric: np.reshape(
                                                    input_quadric, [1, -1])
                                            })
                        # print(val_last[0])
                        self.train_batch_packed(episode_buffer[train_id], sess, self.gamma, val_last[0], train_id)
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

                        # summary = tf.Summary()
                        # summary.value.add(tag='Performance/rewards', simple_value=float(mean_reward))
                        # summary.value.add(tag='Performance/length', simple_value=float(mean_length))
                        # summary.value.add(tag='Performance/values', simple_value=float(mean_value))
                        # summary.value.add(tag='Losses/Value Loss', simple_value=float(val_loss))
                        # summary.value.add(tag='Losses/Prob pred', simple_value=float(pred_prob))
                        # summary.value.add(tag='Losses/Policy Loss', simple_value=float(policy_loss))
                        # summary.value.add(tag='Losses/Grad Norm', simple_value=float(grad_norms))
                        # summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
                        # summary.value.add(tag='Losses/Policy Norm', simple_value=float(p_norm))
                        # summary.value.add(tag='Losses/a0', simple_value=float(a0))
                        #
                        # self.summary_writers[i].add_summary(summary, episodes)
                        # self.summary_writers[i].flush()

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

    load_model = False
    model_path = './model'
    cardgame = env.Env()
    master = CardMaster(cardgame)
    saver = tf.train.Saver(max_to_keep=20)
    with tf.Session() as sess:
        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            master.run(sess, saver, 300)
            # run_game(sess, master)
        else:
            sess.run(tf.global_variables_initializer())
            master.run(sess, saver, 300)
        sess.close()