# from env import Env
import sys

sys.path.insert(0, './build.linux')
from env import Env
# from env_test import Env
import card
from card import action_space, Category, action_space_category
import os, random
import tensorflow as tf
import numpy as np
import threading
import multiprocessing
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import time
# from env_test import get_benchmark
from collections import Counter
import struct
import copy
import random
import argparse
from pyenv import Pyenv
from utils import get_masks, discard_cards, get_mask_alter, \
    give_cards_without_minor, to_char, to_value, get_seq_length, \
    timeblock, gputimeblock, GPUTime, update_params, inference_minor_cards60
from montecarlo import MCTree
from network_RL_v1_1 import CardNetwork
from scheduler import scheduled_run
from tensorflow.python.client import device_lib


def discounted_return(r, gamma):
    r = r.astype(float)
    r_out = np.zeros_like(r)
    val = 0
    for i in reversed(range(r.shape[0])):
        r_out[i] = r[i] + gamma * val
        val = r_out[i]
    return r_out


class CardAgent:
    def __init__(self, name, ngpus):
        self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4, use_locking=True)
        self.name = name
        self.episodes = tf.Variable(0, dtype=tf.int32, name='episodes_' + name, trainable=False)
        self.increment = self.episodes.assign_add(1)
        self.main_network = CardNetwork(self.trainer, name, ngpus)

    def inference_once(self, s, sess):
        is_active = s['control_idx'] == s['idx']
        last_category_idx = s['last_category_idx'] if not is_active else -1
        last_cards_char = s['last_cards'] if not is_active else np.array([])
        last_cards_value = np.array(to_value(last_cards_char)) if not is_active else np.array([])
        last_cards_onehot = card.Card.val2onehot60(last_cards_value)
        curr_cards_char = s['player_cards'][s['idx']]

        state = Pyenv.get_state_static60(s).reshape(1, -1)
        feeddict = (
            (self.main_network.input_state, state),
            (self.main_network.last_outcards, last_cards_onehot.reshape(1, -1)),
        )
        intention = None
        if is_active:
            # first get mask
            decision_mask, response_mask, _, length_mask = get_mask_alter(curr_cards_char, [], False, last_category_idx)

            with gputimeblock('gpu'):
                decision_active_output = scheduled_run(sess, self.main_network.fc_active_decision_output, feeddict)
                # decision_active_output = sess.run(self.main_network.fc_decision_active_output,
                #                                   feed_dict=feeddict)

            # make decision depending on output
            decision_active_output = decision_active_output[0]
            decision_active_output[decision_mask == 0] = -1
            decision_active = np.argmax(decision_active_output)

            active_category_idx = decision_active + 1

            # give actual response
            with gputimeblock('gpu'):
                response_active_output = scheduled_run(sess, self.main_network.fc_active_response_output, feeddict)
                # response_active_output = sess.run(self.main_network.fc_response_active_output,
                #                                   feed_dict=feeddict)

            response_active_output = response_active_output[0]
            response_active_output[response_mask[decision_active] == 0] = -1
            response_active = np.argmax(response_active_output)

            seq_length = 0
            # next sequence length
            if active_category_idx == Category.SINGLE_LINE.value or \
                    active_category_idx == Category.DOUBLE_LINE.value or \
                    active_category_idx == Category.TRIPLE_LINE.value or \
                    active_category_idx == Category.THREE_ONE_LINE.value or \
                    active_category_idx == Category.THREE_TWO_LINE.value:
                with gputimeblock('gpu'):
                    seq_length_output = scheduled_run(sess, self.main_network.fc_active_seq_output, feeddict)
                    # seq_length_output = sess.run(self.main_network.fc_sequence_length_output,
                    #                              feed_dict=feeddict)

                seq_length_output = seq_length_output[0]
                seq_length_output[length_mask[decision_active][response_active] == 0] = -1
                seq_length = np.argmax(seq_length_output) + 1

            # give main cards
            intention = give_cards_without_minor(response_active, last_cards_value, active_category_idx, seq_length)

            # then give minor cards
            if active_category_idx == Category.THREE_ONE.value or \
                    active_category_idx == Category.THREE_TWO.value or \
                    active_category_idx == Category.THREE_ONE_LINE.value or \
                    active_category_idx == Category.THREE_TWO_LINE.value or \
                    active_category_idx == Category.FOUR_TWO.value:
                dup_mask = np.ones([15])
                if seq_length > 0:
                    for i in range(seq_length):
                        dup_mask[intention[0] - 3 + i] = 0
                else:
                    dup_mask[intention[0] - 3] = 0
                intention = np.concatenate([intention, to_value(
                    inference_minor_cards60(active_category_idx, state,
                                          list(curr_cards_char.copy()), sess, self.main_network, seq_length,
                                          dup_mask, to_char(intention))[0])])
        else:
            is_bomb = False
            if len(last_cards_value) == 4 and len(set(last_cards_value)) == 1:
                is_bomb = True
            # print(to_char(last_cards_value), is_bomb, last_category_idx)
            decision_mask, response_mask, bomb_mask, _ = get_mask_alter(curr_cards_char, to_char(last_cards_value),
                                                                        is_bomb, last_category_idx)
            input_single_last, input_pair_last, input_triple_last, input_quadric_last = get_masks(last_cards_char, None)

            # feeddict = (
            #     (self.main_network.training, True),
            #     (self.main_network.input_state, state),
            #     (self.main_network.last_outcards, last_cards_onehot.reshape(1, -1))
            # )
            with gputimeblock('gpu'):
                decision_passive_output, response_passive_output, bomb_passive_output \
                    = scheduled_run(sess, [self.main_network.fc_passive_decision_output,
                                           self.main_network.fc_passive_response_output,
                                           self.main_network.fc_passive_bomb_output], feeddict)
                # decision_passive_output, response_passive_output, bomb_passive_output \
                #     = sess.run([self.main_network.fc_decision_passive_output,
                #                 self.main_network.fc_response_passive_output, self.main_network.fc_bomb_passive_output],
                #                feed_dict=feeddict)

            # print(decision_mask)
            # print(decision_passive_output)
            decision_passive_output = decision_passive_output[0]
            decision_passive_output[decision_mask == 0] = -1
            decision_passive = np.argmax(decision_passive_output)

            if decision_passive == 0:
                intention = np.array([])
            elif decision_passive == 1:
                # print('bomb_mask', bomb_mask)
                bomb_passive_output = bomb_passive_output[0]
                bomb_passive_output[bomb_mask == 0] = -1
                bomb_passive = np.argmax(bomb_passive_output)

                # converting 0-based index to 3-based value
                intention = np.array([bomb_passive + 3] * 4)

            elif decision_passive == 2:
                intention = np.array([16, 17])
            elif decision_passive == 3:
                # print('response_mask', response_mask)
                response_passive_output = response_passive_output[0]
                response_passive_output[response_mask == 0] = -1
                response_passive = np.argmax(response_passive_output)

                intention = give_cards_without_minor(response_passive, last_cards_value, last_category_idx, None)
                if last_category_idx == Category.THREE_ONE.value or \
                        last_category_idx == Category.THREE_TWO.value or \
                        last_category_idx == Category.THREE_ONE_LINE.value or \
                        last_category_idx == Category.THREE_TWO_LINE.value or \
                        last_category_idx == Category.FOUR_TWO.value:
                    dup_mask = np.ones([15])
                    seq_length = get_seq_length(last_category_idx, intention)
                    if seq_length:
                        for i in range(seq_length):
                            dup_mask[intention[0] - 3 + i] = 0
                    else:
                        dup_mask[intention[0] - 3] = 0
                    intention = np.concatenate(
                        [intention, to_value(inference_minor_cards60(last_category_idx, state,
                                                                   list(curr_cards_char.copy()),
                                                                   sess, self.main_network,
                                                                   get_seq_length(last_category_idx, last_cards_value),
                                                                   dup_mask, to_char(intention))[0])])
        return intention

    def get_benchmark(self, sess, num_tests=100):
        env = Pyenv()
        wins = 0
        idx_role_self = 2
        idx_self = -1
        for i in range(num_tests):
            env.reset()
            env.prepare()

            done = False
            while not done:
                s = env.dump_state()
                # use the network
                if idx_role_self == Pyenv.get_role_ID_static(s):
                    idx_self = s['idx']
                    intention = np.array(to_char(self.inference_once(s, sess)))
                else:
                    intention = np.array(to_char(Env.step_auto_static(card.Card.char2color(s['player_cards'][s['idx']]),
                                                                      np.array(to_value(
                                                                          s['last_cards'] if s['control_idx'] != s[
                                                                              'idx'] else [])))))
                r, done = env.step(intention)
            # finished, see who wins
            if idx_self == s['idx']:
                wins += 1
            elif idx_self != s['lord_idx'] and s['idx'] != s['lord_idx']:
                wins += 1
        return wins / num_tests

    def train_batch_sampled(self, buf, batch_size, sess):
        num_of_iters = len(buf) // batch_size + 1
        mean_loss = 0
        mean_var_norm = 0
        mean_grad_norm = 0
        for _ in range(num_of_iters):
            batch_idx = np.random.choice(len(buf), batch_size)
            input_states, input_singles, input_pairs, input_triples, input_quadrics, vals, modes = [
                np.array([buf[i][j] for i in batch_idx]) for j in range(7)]
            passive_decision_input, passive_bomb_input, passive_response_input, active_decision_input, \
            active_response_input, seq_length_input = [np.array([buf[i][7][k] for i in batch_idx])
                                                       for k in ['decision_passive', 'bomb_passive', 'response_passive',
                                                                 'decision_active', 'response_active', 'seq_length']]
            feed_dict = (
                (self.main_network.mode, modes),
                (self.main_network.input_state, input_states),
                (self.main_network.input_single, input_singles),
                (self.main_network.input_pair, input_pairs),
                (self.main_network.input_triple, input_triples),
                (self.main_network.input_quadric, input_quadrics),
                (self.main_network.passive_decision_input, passive_decision_input),
                (self.main_network.passive_response_input, passive_response_input),
                (self.main_network.passive_bomb_input, passive_bomb_input),
                (self.main_network.active_decision_input, active_decision_input),
                (self.main_network.active_response_input, active_response_input),
                (self.main_network.seq_length_input, seq_length_input),
                (self.main_network.value_input, vals)
            )
            _, loss, var_norm, gradient_norm, decision_passive_output, response_passive_output, bomb_passive_output, \
            decision_active_output, response_active_output, seq_length_output \
                = scheduled_run(sess, [self.main_network.optimize, self.main_network.loss, self.main_network.var_norms,
                                       self.main_network.grad_norms, self.main_network.fc_decision_passive_output,
                                       self.main_network.fc_response_passive_output,
                                       self.main_network.fc_bomb_passive_output,
                                       self.main_network.fc_decision_active_output,
                                       self.main_network.fc_response_active_output,
                                       self.main_network.fc_sequence_length_output
                                       ], feed_dict)
            mean_loss += loss
            mean_grad_norm += gradient_norm
            mean_var_norm += var_norm
        mean_loss /= num_of_iters
        mean_var_norm /= num_of_iters
        mean_grad_norm /= num_of_iters
        return mean_loss, mean_var_norm, mean_grad_norm

    def train_batch(self, buf, minor_buf, sess, gamma):
        batch_size = buf.shape[0] + minor_buf.shape[0]
        s, s_last_outcards, rewards, values, seq_length_input, passive_decision_input, passive_response_input, \
        passive_bomb_input, active_decision_input, active_response_input, mode = [buf[:, i] for i in range(11)]

        s = np.vstack(s)
        s_last_outcards = np.vstack(s_last_outcards)

        val_truth = discounted_return(rewards, gamma)
        val_pred_plus = np.append(values, 0)
        td0 = rewards + gamma * val_pred_plus[1:] - val_pred_plus[:-1]
        advantages = discounted_return(td0, gamma)
        feed_dict = (
            (self.main_network.input_state, s),
            (self.main_network.last_outcards, s_last_outcards),
            (self.main_network.minor_type, np.zeros([buf.shape[0]])),
            (self.main_network.mode, mode),
            (self.main_network.passive_decision_input, passive_decision_input),
            (self.main_network.passive_response_input, passive_response_input),
            (self.main_network.passive_bomb_input, passive_bomb_input),
            (self.main_network.active_decision_input, active_decision_input),
            (self.main_network.active_response_input, active_response_input),
            (self.main_network.minor_response_input, active_response_input), # not used
            (self.main_network.seq_length_input, seq_length_input),
            (self.main_network.advantages_input, advantages),
            (self.main_network.value_input, val_truth)
        )
        _, policy_loss, value_loss, gradient_norm \
            = scheduled_run(sess, [self.main_network.optimize, self.main_network.policy_loss, self.main_network.value_loss,
                                   self.main_network.gradient_norms], feed_dict)
        # minor target
        if minor_buf.size > 0:
            s_minor, minor_response_input, minor_type, minor_idx = [minor_buf[:, i] for i in range(4)]
            s_minor = np.vstack(s_minor)
            rows = minor_buf.shape[0]
            feed_dict = (
                (self.main_network.mode, np.ones([rows]) * 5),
                (self.main_network.input_state, s_minor),
                (self.main_network.last_outcards, np.zeros([rows, 60])),
                (self.main_network.minor_type, minor_type),
                (self.main_network.passive_decision_input, np.zeros([rows])),
                (self.main_network.passive_response_input, np.zeros([rows])),
                (self.main_network.passive_bomb_input, np.zeros([rows])),
                (self.main_network.active_decision_input, np.zeros([rows])),
                (self.main_network.active_response_input, np.zeros([rows])),
                (self.main_network.minor_response_input, minor_response_input),
                (self.main_network.seq_length_input, np.zeros([rows])),
                (self.main_network.advantages_input, advantages[minor_idx.astype(np.int)]),
                (self.main_network.value_input, np.zeros([rows]))
            )
            _, policy_loss_minor, value_loss_minor, gradient_norm_minor \
                = scheduled_run(sess, [self.main_network.optimize, self.main_network.policy_loss, self.main_network.value_loss,
                                       self.main_network.gradient_norms], feed_dict)
            policy_loss += policy_loss_minor
            value_loss += value_loss_minor
            gradient_norm += gradient_norm_minor
        # calculate mean w.r.t batch
        return policy_loss / batch_size, value_loss / batch_size, gradient_norm / batch_size


class CardMaster:
    def __init__(self, name, ngpus, global_episodes):
        self.temp = 1
        self.start_temp = 1
        self.end_temp = 0.2
        self.gamma = 1.0
        self.name = name
        self.env = Pyenv()
        self.sess = None

        self.episode_rewards = []
        self.episode_length = []
        self.episode_mean_values = []
        self.episode_policy_loss = []
        self.episode_value_loss = []
        self.episode_gradient_norms = []
        self.summary_writer = tf.summary.FileWriter(name)

        self.agent = CardAgent(name, ngpus)

        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.update_local_from_global_ops = update_params('global', self.name)

    def train_batch_sampled(self, buf, batch_size, sess):
        return self.agent.train_batch_sampled(buf, batch_size, sess)

    def train_batch(self, buf, minor_buf, sess):
        return self.agent.train_batch(np.array(buf), np.array(minor_buf), sess, self.gamma)

    def print_benchmarks(self, sess, num_tests=100):
        tf.logging.info("BENCHMARK %s: %f" % (self.agent.name, self.agent.get_benchmark(sess, num_tests)))
        # print("%s: %f" % (self.agents[1].name, self.agents[1].get_benchmark(sess)))

    # train three agents simultaneously
    def run(self, sess, saver, coord, total_episodes, global_network):
        self.sess = sess
        with sess.as_default():
            temp_decay = (self.end_temp - self.start_temp) / total_episodes
            precision_eps = 1e-7
            episode_count = sess.run(self.global_episodes)
            while not coord.should_stop():
                if episode_count >= total_episodes:
                    break

                sess.run(self.update_local_from_global_ops)
                self.env.reset()
                self.env.prepare()
                lord_idx = self.env.lord_idx
                tf.logging.info("%s: episode %d" % (self.name, episode_count))
                episode_buffer = []
                episode_minor_buffer = []
                episode_values = []
                episode_reward = 0
                episode_steps = 0

                max_length = 1000
                for l in range(max_length):
                    assert self.env.idx == lord_idx
                    s = self.env.get_state60().reshape(1, -1)

                    curr_cards_char = self.env.get_handcards()
                    last_cards_char = self.env.get_last_outcards()
                    last_cards_value = np.array(to_value(last_cards_char) if last_cards_char is not None else [])
                    last_category_idx = self.env.get_last_category_idx()
                    last_cards_onehot = card.Card.val2onehot60(last_cards_value).reshape(1, -1)

                    feed_dict = (
                        (self.agent.main_network.input_state, s),
                        (self.agent.main_network.last_outcards, last_cards_onehot)
                    )
                    # feed forward, exploring
                    decision_passive_output, response_passive_output, bomb_passive_output, \
                    decision_active_output, response_active_output, seq_length_output, val_output \
                        = scheduled_run(sess, [self.agent.main_network.fc_passive_decision_output,
                                         self.agent.main_network.fc_passive_response_output,
                                         self.agent.main_network.fc_passive_bomb_output,
                                         self.agent.main_network.fc_active_decision_output,
                                         self.agent.main_network.fc_active_response_output,
                                         self.agent.main_network.fc_active_seq_output,
                                         self.agent.main_network.fc_value_output], feed_dict
                                        )

                    intention = None
                    seq_length_input = -1
                    passive_decision_input = -1
                    passive_response_input = -1
                    passive_bomb_input = -1
                    active_decision_input = -1
                    active_response_input = -1
                    mode = -1
                    if last_cards_value.size > 0:
                        is_bomb = False
                        if len(last_cards_value) == 4 and len(set(last_cards_value)) == 1:
                            is_bomb = True
                        decision_mask, response_mask, bomb_mask, _ = get_mask_alter(curr_cards_char, last_cards_char,
                                                                                    is_bomb, last_category_idx)
                        decision_passive_output = (decision_passive_output[0] + precision_eps) * decision_mask
                        decision_passive = np.random.choice(4, 1, p=decision_passive_output / decision_passive_output.sum())[0]

                        # save to buffer
                        passive_decision_input = decision_passive
                        if decision_passive == 0:
                            mode = 0
                            intention = np.array([])
                        elif decision_passive == 1:
                            bomb_passive_output = (bomb_passive_output[0] + precision_eps) * bomb_mask
                            mode = 1
                            # save to buffer
                            passive_bomb_input = np.random.choice(13, 1, p=bomb_passive_output / bomb_passive_output.sum())[0]

                            # converting 0-based index to 3-based value
                            intention = np.array([passive_bomb_input + 3] * 4)
                        elif decision_passive == 2:
                            mode = 0
                            intention = np.array([16, 17])
                        elif decision_passive == 3:
                            mode = 2
                            response_passive_output = (response_passive_output[0] + precision_eps) * response_mask

                            # save to buffer
                            passive_response_input = np.random.choice(15, 1, p=response_passive_output / response_passive_output.sum())[0]

                            intention = give_cards_without_minor(passive_response_input, last_cards_value, last_category_idx,
                                                                None)
                            minor_type = 0
                            if last_category_idx == Category.THREE_ONE.value or \
                                    last_category_idx == Category.THREE_TWO.value or \
                                    last_category_idx == Category.THREE_ONE_LINE.value or \
                                    last_category_idx == Category.THREE_TWO_LINE.value or \
                                    last_category_idx == Category.FOUR_TWO.value:
                                if last_category_idx == Category.THREE_TWO.value or last_category_idx == Category.THREE_TWO_LINE.value:
                                    minor_type = 1
                                dup_mask = np.ones([15])
                                seq_length = get_seq_length(last_category_idx, intention)
                                if seq_length:
                                    for i in range(seq_length):
                                        dup_mask[intention[0] - 3 + i] = 0
                                else:
                                    dup_mask[intention[0] - 3] = 0
                                minor_cards_val, inter_states, inter_outputs = \
                                    inference_minor_cards60(last_category_idx, s.copy(), list(curr_cards_char.copy()),
                                                          sess, self.agent.main_network,
                                                          get_seq_length(last_category_idx, last_cards_value),
                                                          dup_mask, to_char(intention))
                                minor_cards_val = to_value(minor_cards_val)
                                for i in range(len(inter_states)):
                                    episode_minor_buffer.append([
                                                                           inter_states[i],
                                                                           inter_outputs[i],
                                                                           minor_type,
                                        len(episode_buffer) - 1])

                                intention = np.concatenate([intention, minor_cards_val])
                    else:
                        mode = 3
                        decision_mask, response_mask, _, length_mask = get_mask_alter(curr_cards_char, [], False,
                                                                                      last_category_idx)
                        # first the decision with argmax applied
                        decision_active_output = (decision_active_output[0] + precision_eps) * decision_mask

                        # save to buffer
                        active_decision_input = \
                        np.random.choice(13, 1, p=decision_active_output / decision_active_output.sum())[0]

                        decision_active = active_decision_input

                        # then convert 0-based decision_active_output to 1-based (empty eliminated) category idx
                        active_category_idx = active_decision_input + 1

                        # then the actual response to represent card value
                        response_active_output = (response_active_output[0] + precision_eps) * response_mask[decision_active]

                        # save to buffer
                        active_response_input = \
                        np.random.choice(15, 1, p=response_active_output / response_active_output.sum())[0]

                        # get length mask
                        seq_length_output = (seq_length_output[0] + precision_eps) * length_mask[decision_active][active_response_input]

                        # save to buffer
                        seq_length_input = np.random.choice(12, 1, p=seq_length_output / seq_length_output.sum())[0]

                        # seq length only has OFFSET 1 from 0-11 to 1-12 ('3' - 'A')
                        seq_length = int(seq_length_input + 1)

                        if active_category_idx == Category.SINGLE_LINE.value or \
                                active_category_idx == Category.DOUBLE_LINE.value or \
                                active_category_idx == Category.TRIPLE_LINE.value or \
                                active_category_idx == Category.THREE_ONE_LINE.value or \
                                active_category_idx == Category.THREE_TWO_LINE.value:
                            mode = 4

                        # give main cards
                        intention = give_cards_without_minor(active_response_input, last_cards_value, active_category_idx,
                                                             seq_length)

                        # then give minor cards
                        minor_type = 0
                        if active_category_idx == Category.THREE_ONE.value or \
                                active_category_idx == Category.THREE_TWO.value or \
                                active_category_idx == Category.THREE_ONE_LINE.value or \
                                active_category_idx == Category.THREE_TWO_LINE.value or \
                                active_category_idx == Category.FOUR_TWO.value:
                            if last_category_idx == Category.THREE_TWO.value or last_category_idx == Category.THREE_TWO_LINE.value:
                                minor_type = 1
                            dup_mask = np.ones([15])
                            if seq_length > 0:
                                for i in range(seq_length):
                                    dup_mask[intention[0] - 3 + i] = 0
                            else:
                                dup_mask[intention[0] - 3] = 0

                            minor_cards_val, inter_states, inter_outputs = \
                                inference_minor_cards60(active_category_idx, s.copy(), list(curr_cards_char.copy()),
                                                      sess, self.agent.main_network,
                                                      seq_length,
                                                      dup_mask, to_char(intention))
                            minor_cards_val = to_value(minor_cards_val)
                            for i in range(len(inter_states)):
                                episode_minor_buffer.append([
                                                                       inter_states[i],
                                                                       inter_outputs[i],
                                                                       minor_type,
                                    len(episode_buffer) - 1])

                            intention = np.concatenate([intention, minor_cards_val])

                    # print(train_id, 'single step')
                    # print(self.env.idx, ": ", to_char(intention))
                    r, done = self.env.step(np.array(to_char(intention)))

                    episode_reward = r
                    episode_steps += 1
                    episode_values.append(val_output[0][0])
                    episode_buffer.append(
                        [s[0], last_cards_onehot[0], r, val_output[0][0], seq_length_input, passive_decision_input, passive_response_input,
                         passive_bomb_input, active_decision_input, active_response_input, mode])
                    # self.train_batch_sampled([episode_buffer[train_id]], 8)

                    if done:
                        logs = self.train_batch(episode_buffer, episode_minor_buffer, sess)
                        self.episode_policy_loss.append(logs[0])
                        self.episode_value_loss.append(logs[1])
                        self.episode_gradient_norms.append(logs[2])

                        # then sample buffer for training
                        # logs = self.train_batch_sampled([episode_buffer[i] for i in range(3)], 32, sess)
                        break

                    # step for farmers
                    finished = False
                    for _ in range(2):
                        intention = np.array(
                            to_char(Env.step_auto_static(card.Card.char2color(self.env.player_cards[self.env.idx]),
                                                         np.array(to_value(self.env.last_cards if self.env.control_idx != self.env.idx else [])))))
                        r, done = self.env.step(intention)
                        if done:
                            episode_reward = -r
                            episode_buffer[-1][2] = -r
                            logs = self.train_batch(episode_buffer, episode_minor_buffer, sess)
                            self.episode_policy_loss.append(logs[0])
                            self.episode_value_loss.append(logs[1])
                            self.episode_gradient_norms.append(logs[2])
                            finished = True
                            break
                    if finished:
                        break

                # self.episode_mean_values[i].append(np.mean(episode_values[i]))
                self.episode_length.append(episode_steps)
                self.episode_rewards.append(episode_reward)
                self.episode_mean_values.append(np.mean(episode_values))

                update_rate = 5
                if episode_count % update_rate == 0 and episode_count > 0:
                    mean_reward = np.mean(self.episode_rewards[-update_rate:])
                    mean_length = np.mean(self.episode_length[-update_rate:])
                    mean_value = np.mean(self.episode_mean_values[-update_rate:])
                    mean_policy_loss = np.mean(self.episode_policy_loss[-update_rate:])
                    mean_value_loss = np.mean(self.episode_value_loss[-update_rate:])
                    mean_grad_norms = np.mean(self.episode_gradient_norms[-update_rate:])

                    summary = tf.Summary()
                    # TODO: add more summary with value loss
                    summary.value.add(tag='Performance/rewards', simple_value=float(mean_reward))
                    summary.value.add(tag='Performance/length', simple_value=float(mean_length))
                    summary.value.add(tag='Performance/values', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/policy loss', simple_value=float(mean_policy_loss))
                    summary.value.add(tag='Losses/value loss', simple_value=float(mean_value_loss))
                    summary.value.add(tag='Losses/grad norm', simple_value=float(mean_grad_norms))

                    # summary.value.add(tag='Losses/Policy Norm', simple_value=float(p_norm))
                    # summary.value.add(tag='Losses/a0', simple_value=float(a0))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                if self.name == 'agent_0':
                    sess.run(self.increment)
                    if episode_count % 100 == 0 and episode_count > 0:
                        summary = tf.Summary()
                        non_lstm_weight_norm, lstm_weight_norm = sess.run(
                            [global_network.non_lstm_weight_norms, global_network.lstm_weight_norms])
                        summary.value.add(tag='Norm/weight norm', simple_value=float(non_lstm_weight_norm))
                        summary.value.add(tag='Norm/lstm weight norm', simple_value=float(lstm_weight_norm))
                        self.summary_writer.add_summary(summary, episode_count)
                        self.summary_writer.flush()
                    if episode_count % 300 == 0:
                        if episode_count > 0:
                            saver.save(sess, "./Model/a3c_1.4/model", global_step=episode_count)
                            tf.logging.info('saved model')
                            self.print_benchmarks(sess, 100)
                episode_count += 1


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def name_in_checkpoint(v):
    # name = v.name.replace('global', 'network')
    # if 'beta' in v.name:
    #     suffix = v.name.split('/')[-1].split(':')[0].split('_', 1)[1]
    #     if '_' in suffix:
    #         new_suffix = suffix.split('_')[-1] + '_' + suffix.split('_')[0]
    #     else:
    #         new_suffix = suffix
    #     # print(name.replace(name.split('/')[-1], new_suffix + ':0'))
    #     return name.replace(name.split('/')[-1], new_suffix)
    return v.name.replace(':0', '')


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
    tf.logging.set_verbosity(tf.logging.INFO)
    global_agent = CardAgent('global', 1)

    variables_to_save = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
    saver = tf.train.Saver(variables_to_save, max_to_keep=100)
    variables_to_restore = {name_in_checkpoint(v): v for v in variables_to_save}
    restorer = tf.train.Saver(variables_to_restore)

    global_episode = tf.Variable(10000, dtype=tf.int32, name='global_episodes', trainable=False)
    variables_to_save.append(global_episode)
    # variables_to_restore['global_episodes'] = global_episode

    num_agents = multiprocessing.cpu_count()
    num_agents = 12
    print('num of cpus ', num_agents)
    agents = []
    for ag in range(num_agents):
        agents.append(CardMaster('agent_%d' % ag, len(get_available_gpus()), global_episode))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        tf.logging.info('loading pretrained RL model....')
        restorer.restore(sess, "./Model/a3c_1.4/model-9900")
        tf.logging.info('loaded pretrained RL model.')
        # tf.logging.info('BENCHMARK(PRETRAINED): {}'.format(global_agent.get_benchmark(sess)))

        threads = []
        for agent in agents:
            agent_run = lambda: agent.run(sess, saver, coord, 1e5, global_agent.main_network)
            t = threading.Thread(target=(agent_run))
            t.start()
            # time.sleep(0.5)
            threads.append(t)
        coord.join(threads)
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     saver.restore(sess, "./Model/SL_lite/model-19800")
    #     print('benchmark', agent.get_benchmark(sess))
    # trainer = tf.train.AdadeltaOptimizer(learning_rate=1e-4)
    # main_network = CardNetwork(trainer, 'global', 1)
    # available_gpus = len(get_available_gpus())
    # agent_network = CardNetwork(trainer, 'agent', available_gpus)
    # agent_network2 = CardNetwork(trainer, 'agent2', available_gpus)
    # parser = argparse.ArgumentParser(description='fight the lord')
    # # parser.add_argument('--b', type=int, help='batch size', default=32)
    # parser.add_argument('--epoches_train', type=int, help='num of epochs to train', default=1)
    # parser.add_argument('--epoches_test', type=int, help='num of epochs to test', default=0)
    # # parser.add_argument('--train', type=bool, help='whether to train', default=True)
    #
    # args = parser.parse_args(sys.argv[1:])
    # epoches_train = args.epoches_train
    # epoches_test = args.epoches_test
    #
    # a_dim = len(action_space)
    #
    # load_model = True
    # model_path = './model'
    # master = CardMaster()
    # # variables_to_restore = slim.get_variables(scope='agent0')
    # variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent0')
    # variables_to_restore = {name_in_checkpoint(v): v for v in variables_to_restore if "value_output" not in v.name}
    #
    # # vars_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent0')
    # # for v in vars_train:
    # #     print(v.name + ':', tf.shape(v))
    # saver = tf.train.Saver(variables_to_restore, max_to_keep=100)
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #     # variables_names = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent0')]
    #     # sess.run(tf.global_variables_initializer())
    #     # values = sess.run(variables_names)
    #     # for k, v in zip(variables_names, values):
    #     #     print("Variable: ", k)
    #     #     print("Shape: ", v.shape)
    #     sess.run(tf.global_variables_initializer())
    #     saver.restore(sess, "./Model/accuracy_fake_minor/model-9500")
    #     master.update_params_from_agent0(sess)
    #     master.print_benchmarks(sess)
    #     master.run(sess, saver, 300)
        # master.print_benchmarks(sess)
        # print v
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    #     if load_model:
    #         print('Loading Model...')
    #         # ckpt = tf.train.get_checkpoint_state(model_path)
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #         print('Loaded model.')
    #         # master.run(sess, saver, 300)
    #         # run_game(sess, master)
    #     else:
    #         sess.run(tf.global_variables_initializer())
    # master.run(sess, saver, 300)

