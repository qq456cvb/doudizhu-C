# from env import Env
import sys

sys.path.insert(0, './build/Release')
from env import Env
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
from pyenv import Pyenv
from utils import get_masks, discard_cards, get_mask_alter, \
    give_cards_without_minor, to_char, to_value, inference_minor_cards, get_seq_length, \
    timeblock, gputimeblock, GPUTime, update_params
from montecarlo import MCTree
from network_a3c import CardNetwork
from scheduler import scheduled_run


def discounted_return(r, gamma):
    r = r.astype(float)
    r_out = np.zeros_like(r)
    val = 0
    for i in reversed(range(r.shape[0])):
        r_out[i] = r[i] + gamma * val
        val = r_out[i]
    return r_out


class CardAgent:
    def __init__(self, name):
        self.trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.name = name
        self.episodes = tf.Variable(0, dtype=tf.int32, name='episodes_' + name, trainable=False)
        self.increment = self.episodes.assign_add(1)
        self.main_network = CardNetwork(54 * 6, self.trainer, self.name, 2)

    def inference_once(self, s, sess):
        is_active = s['control_idx'] == s['idx']
        last_category_idx = s['last_category_idx'] if not is_active else -1
        last_cards_char = s['last_cards'] if not is_active else np.array([])
        last_cards_value = np.array(to_value(last_cards_char)) if not is_active else np.array([])
        curr_cards_char = s['player_cards'][s['idx']]

        input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, last_cards_char)

        state = Pyenv.get_state_static(s).reshape(1, -1)
        feeddict = (
            (self.main_network.training, True),
            (self.main_network.input_state, state),
            (self.main_network.input_single, input_single.reshape(1, -1)),
            (self.main_network.input_pair, input_pair.reshape(1, -1)),
            (self.main_network.input_triple, input_triple.reshape(1, -1)),
            (self.main_network.input_quadric, input_quadric.reshape(1, -1))
        )
        intention = None
        if is_active:
            # first get mask
            decision_mask, response_mask, _, length_mask = get_mask_alter(curr_cards_char, [], False, last_category_idx)

            with gputimeblock('gpu'):
                decision_active_output = scheduled_run(sess, self.main_network.fc_decision_active_output, feeddict)
                # decision_active_output = sess.run(self.main_network.fc_decision_active_output,
                #                                   feed_dict=feeddict)

            # make decision depending on output
            decision_active_output = decision_active_output[0]
            decision_active_output[decision_mask == 0] = -1
            decision_active = np.argmax(decision_active_output)

            active_category_idx = decision_active + 1

            # give actual response
            with gputimeblock('gpu'):
                response_active_output = scheduled_run(sess, self.main_network.fc_response_active_output, feeddict)
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
                    seq_length_output = scheduled_run(sess, self.main_network.fc_sequence_length_output, feeddict)
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
                    inference_minor_cards(active_category_idx, state,
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

            feeddict = (
                (self.main_network.training, True),
                (self.main_network.input_state, state),
                (self.main_network.input_single, input_single.reshape(1, -1)),
                (self.main_network.input_pair, input_pair.reshape(1, -1)),
                (self.main_network.input_triple, input_triple.reshape(1, -1)),
                (self.main_network.input_quadric, input_quadric.reshape(1, -1)),
                (self.main_network.input_single_last, input_single_last.reshape(1, -1)),
                (self.main_network.input_pair_last, input_pair_last.reshape(1, -1)),
                (self.main_network.input_triple_last, input_triple_last.reshape(1, -1)),
                (self.main_network.input_quadric_last, input_quadric_last.reshape(1, -1)),
            )
            with gputimeblock('gpu'):
                decision_passive_output, response_passive_output, bomb_passive_output \
                    = scheduled_run(sess, [self.main_network.fc_decision_passive_output,
                                           self.main_network.fc_response_passive_output,
                                           self.main_network.fc_bomb_passive_output], feeddict)
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
                        [intention, to_value(inference_minor_cards(last_category_idx, state,
                                                                   list(curr_cards_char.copy()),
                                                                   sess, self.main_network,
                                                                   get_seq_length(last_category_idx, last_cards_value),
                                                                   dup_mask, to_char(intention))[0])])
        return intention

    def get_benchmark(self, sess):
        num_tests = 100
        env = Pyenv()
        wins = 0
        idx_role_self = int(self.name[-1]) + 1
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
                (self.main_network.training, True),
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
        s, rewards, values, input_single, input_pair, input_triple, \
        input_quadric, input_single_last, input_pair_last, input_triple_last, \
        input_quadric_last, seq_length_input, passive_decision_input, passive_response_input, \
        passive_bomb_input, active_decision_input, active_response_input, mode = [buf[:, i] for i in range(18)]

        s = np.vstack(s)
        input_single = np.vstack(input_single)
        input_pair = np.vstack(input_pair)
        input_triple = np.vstack(input_triple)
        input_quadric = np.vstack(input_quadric)
        input_single_last = np.vstack(input_single_last)
        input_pair_last = np.vstack(input_pair_last)
        input_triple_last = np.vstack(input_triple_last)
        input_quadric_last = np.vstack(input_quadric_last)

        val_truth = discounted_return(rewards, gamma)
        val_pred_plus = np.append(values, 0)
        td0 = rewards + gamma * val_pred_plus[1:] - val_pred_plus[:-1]
        advantages = discounted_return(td0, gamma)
        feed_dict = (
            (self.main_network.training, True),
            (self.main_network.mode, mode),
            (self.main_network.input_state, s),
            (self.main_network.input_single, input_single),
            (self.main_network.input_pair, input_pair),
            (self.main_network.input_triple, input_triple),
            (self.main_network.input_quadric, input_quadric),
            (self.main_network.input_single_last, input_single_last),
            (self.main_network.input_pair_last, input_pair_last),
            (self.main_network.input_triple_last, input_triple_last),
            (self.main_network.input_quadric_last, input_quadric_last),
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
        _, loss, var_norm, gradient_norm \
            = scheduled_run(sess, [self.main_network.optimize, self.main_network.loss, self.main_network.var_norms,
                                   self.main_network.grad_norms], feed_dict)
        # minor target
        if minor_buf.size > 0:
            s_minor, input_single, input_pair, input_triple, input_quadric, minor_response_input, minor_idx = [minor_buf[:, i] for i in range(7)]
            s_minor = np.vstack(s_minor)
            input_single = np.vstack(input_single)
            input_pair = np.vstack(input_pair)
            input_triple = np.vstack(input_triple)
            input_quadric = np.vstack(input_quadric)
            rows = minor_buf.shape[0]
            feed_dict = (
                (self.main_network.training, True),
                (self.main_network.mode, np.ones([rows]) * 5),
                (self.main_network.input_state, s_minor),
                (self.main_network.input_single, input_single),
                (self.main_network.input_pair, input_pair),
                (self.main_network.input_triple, input_triple),
                (self.main_network.input_quadric, input_quadric),
                (self.main_network.input_single_last, np.zeros([rows, 15])),
                (self.main_network.input_pair_last, np.zeros([rows, 13])),
                (self.main_network.input_triple_last, np.zeros([rows, 13])),
                (self.main_network.input_quadric_last, np.zeros([rows, 13])),
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
            _, loss_minor, var_norm_minor, gradient_norm_minor \
                = scheduled_run(sess, [self.main_network.optimize, self.main_network.loss, self.main_network.var_norms,
                                       self.main_network.grad_norms], feed_dict)
            return loss, var_norm, gradient_norm, loss_minor, var_norm_minor, gradient_norm_minor

        return loss, var_norm, gradient_norm, 0, 0, 0


class CardMaster:
    def __init__(self):
        self.temp = 1
        self.start_temp = 1
        self.end_temp = 0.2
        self.gamma = 0.99
        self.name = 'global'
        self.env = Pyenv()
        self.sess = None

        self.train_intervals = 10

        self.episode_rewards = [[] for i in range(3)]
        self.episode_length = [[] for i in range(3)]
        self.episode_mean_values = [[] for i in range(3)]
        self.summary_writers = [tf.summary.FileWriter("train_agent%d" % i) for i in range(3)]

        self.agents = [CardAgent('agent%d' % i) for i in range(3)]

        self.global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        self.increment = self.global_episodes.assign_add(1)

        self.update_ops = update_params('agent0', 'agent1')
        self.update_ops += update_params('agent0', 'agent2')

    def train_batch_sampled(self, buf, batch_size, sess):
        logs = []
        for i in range(len(buf)):
            logs.append(self.agents[i].train_batch_sampled(buf[i], batch_size, sess))
        return logs

    def train_batch(self, buf, minor_buf, sess):
        logs = []
        for i in range(len(buf)):
            logs.append(self.agents[i].train_batch(np.array(buf[i]), np.array(minor_buf[i]), sess, self.gamma))
        return logs

    def print_benchmarks(self, sess):
        for agent in self.agents:
            print("%s: %f" % (agent.name, agent.get_benchmark(sess)))
        # print("%s: %f" % (self.agents[1].name, self.agents[1].get_benchmark(sess)))

    def update_params_from_agent0(self, sess):
        sess.run(self.update_ops)

    # train three agents simultaneously
    def run(self, sess, saver, max_episode_length):
        self.sess = sess
        with sess.as_default():
            global_episodes = sess.run(self.global_episodes)
            total_episodes = 10001
            temp_decay = (self.end_temp - self.start_temp) / total_episodes
            precision_eps = 1e-7
            while global_episodes < total_episodes:
                self.env.reset()
                self.env.prepare()
                lord_idx = self.env.lord_idx
                print("episode %d" % global_episodes)
                episode_buffer = [[] for _ in range(3)]
                episode_minor_buffer = [[] for _ in range(3)]
                episode_values = [[] for _ in range(3)]
                episode_reward = [0, 0, 0]
                episode_steps = [0, 0, 0]

                logs = []
                for l in range(max_episode_length):
                    s = self.env.get_state().reshape(1, -1)
                    # time.sleep(1)
                    # # map 1-3 to 0-2
                    train_id = self.env.get_role_ID() - 1

                    curr_cards_char = self.env.get_handcards()
                    last_cards_char = self.env.get_last_outcards()
                    last_cards_value = np.array(to_value(last_cards_char) if last_cards_char is not None else [])
                    last_category_idx = self.env.get_last_category_idx()

                    input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, last_cards_char)
                    input_single_last, input_pair_last, input_triple_last, input_quadric_last = get_masks(
                        last_cards_char, None)

                    feed_dict = (
                        (self.agents[train_id].main_network.training, False),
                        (self.agents[train_id].main_network.input_state, s),
                        (self.agents[train_id].main_network.input_single, np.reshape(input_single, [1, -1])),
                        (self.agents[train_id].main_network.input_pair, np.reshape(input_pair, [1, -1])),
                        (self.agents[train_id].main_network.input_triple, np.reshape(input_triple, [1, -1])),
                        (self.agents[train_id].main_network.input_quadric, np.reshape(input_quadric, [1, -1])),
                        (self.agents[train_id].main_network.input_single_last, np.reshape(input_single_last, [1, -1])),
                        (self.agents[train_id].main_network.input_pair_last, np.reshape(input_pair_last, [1, -1])),
                        (self.agents[train_id].main_network.input_triple_last, np.reshape(input_triple_last, [1, -1])),
                        (self.agents[train_id].main_network.input_quadric_last, np.reshape(input_quadric_last, [1, -1]))
                    )
                    # forward feed, exploring
                    decision_passive_output, response_passive_output, bomb_passive_output, \
                    decision_active_output, response_active_output, seq_length_output, val_output \
                        = scheduled_run(sess, [self.agents[train_id].main_network.fc_decision_passive_output,
                                         self.agents[train_id].main_network.fc_response_passive_output,
                                         self.agents[train_id].main_network.fc_bomb_passive_output,
                                         self.agents[train_id].main_network.fc_decision_active_output,
                                         self.agents[train_id].main_network.fc_response_active_output,
                                         self.agents[train_id].main_network.fc_sequence_length_output,
                                         self.agents[train_id].main_network.fc_value_output], feed_dict
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
                                minor_cards_val, inter_states, inter_masks, inter_outputs = \
                                    inference_minor_cards(last_category_idx, s.copy(), list(curr_cards_char.copy()),
                                                          sess, self.agents[train_id].main_network,
                                                          get_seq_length(last_category_idx, last_cards_value),
                                                          dup_mask, to_char(intention))
                                minor_cards_val = to_value(minor_cards_val)
                                for i in range(len(inter_states)):
                                    episode_minor_buffer[train_id].append([
                                                                           inter_states[i], inter_masks[i][0],
                                                                           inter_masks[i][1],
                                                                           inter_masks[i][2], inter_masks[i][3],
                                                                           inter_outputs[i],
                                        len(episode_buffer[train_id]) - 1])

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

                            minor_cards_val, inter_states, inter_masks, inter_outputs = \
                                inference_minor_cards(active_category_idx, s.copy(), list(curr_cards_char.copy()),
                                                      sess, self.agents[train_id].main_network,
                                                      seq_length,
                                                      dup_mask, to_char(intention))
                            minor_cards_val = to_value(minor_cards_val)
                            for i in range(len(inter_states)):
                                episode_minor_buffer[train_id].append([
                                                                       inter_states[i], inter_masks[i][0],
                                                                       inter_masks[i][1],
                                                                       inter_masks[i][2], inter_masks[i][3],
                                                                       inter_outputs[i],
                                    len(episode_buffer[train_id]) - 1])

                            intention = np.concatenate([intention, minor_cards_val])

                    # print(train_id, 'single step')
                    print(self.env.idx, ": ", to_char(intention))
                    r, done = self.env.step(np.array(to_char(intention)))


                    episode_reward[train_id] += r
                    episode_steps[train_id] += 1
                    episode_buffer[train_id].append(
                        [s[0], r, val_output[0], input_single, input_pair, input_triple,
                         input_quadric, input_single_last, input_pair_last, input_triple_last,
                         input_quadric_last, seq_length_input, passive_decision_input, passive_response_input,
                         passive_bomb_input, active_decision_input, active_response_input, mode])
                    # self.train_batch_sampled([episode_buffer[train_id]], 8)

                    if done:
                        for i in range(3):
                            if i == train_id:
                                continue
                            if train_id == lord_idx:
                                episode_buffer[i][-1][1] = -r
                                episode_reward[i] -= r
                            elif i == lord_idx:
                                episode_buffer[i][-1][1] = -r
                                episode_reward[i] -= r
                            else:
                                episode_buffer[i][-1][1] = r
                                episode_reward[i] += r

                        logs = self.train_batch(episode_buffer, episode_minor_buffer, sess)

                        # then sample buffer for training
                        # logs = self.train_batch_sampled([episode_buffer[i] for i in range(3)], 32, sess)
                        break

                for i in range(3):
                    # self.episode_mean_values[i].append(np.mean(episode_values[i]))
                    self.episode_length[i].append(episode_steps[i])
                    self.episode_rewards[i].append(episode_reward[i])

                    episodes = sess.run(self.agents[i].episodes)
                    sess.run(self.agents[i].increment)

                    update_rate = 5
                    if episodes % update_rate == 0 and episodes > 0:
                        mean_reward = np.mean(self.episode_rewards[i][-update_rate:])
                        mean_length = np.mean(self.episode_length[i][-update_rate:])
                        # mean_value = np.mean(self.episode_mean_values[i][-update_rate:])

                        summary = tf.Summary()
                        # TODO: add more summary with value loss
                        summary.value.add(tag='Performance/rewards', simple_value=float(mean_reward))
                        summary.value.add(tag='Performance/length', simple_value=float(mean_length))
                        # summary.value.add(tag='Performance/values', simple_value=float(mean_value))
                        summary.value.add(tag='Losses/Loss', simple_value=float(logs[i][0]))
                        # summary.value.add(tag='Losses/Prob pred', simple_value=float(pred_prob))
                        # summary.value.add(tag='Losses/Policy Loss', simple_value=float(policy_loss))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(logs[i][1]))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(logs[i][2]))
                        # summary.value.add(tag='Losses/Policy Norm', simple_value=float(p_norm))
                        # summary.value.add(tag='Losses/a0', simple_value=float(a0))
                        self.summary_writers[i].add_summary(summary, episodes)
                        self.summary_writers[i].flush()

                global_episodes += 1
                sess.run(self.increment)
                if global_episodes % 100 == 0:
                    saver.save(sess, './model' + '/model-' + str(global_episodes) + '.cptk')
                    print("Saved Model")

                # self.env.end()


def name_in_checkpoint(var):
    return var.op.name.replace("agent0", "SLNetwork")


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

    parser = argparse.ArgumentParser(description='fight the lord')
    # parser.add_argument('--b', type=int, help='batch size', default=32)
    parser.add_argument('--epoches_train', type=int, help='num of epochs to train', default=1)
    parser.add_argument('--epoches_test', type=int, help='num of epochs to test', default=0)
    # parser.add_argument('--train', type=bool, help='whether to train', default=True)

    args = parser.parse_args(sys.argv[1:])
    epoches_train = args.epoches_train
    epoches_test = args.epoches_test

    a_dim = len(action_space)

    load_model = True
    model_path = './model'
    master = CardMaster()
    # variables_to_restore = slim.get_variables(scope='agent0')
    variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent0')
    variables_to_restore = {name_in_checkpoint(v): v for v in variables_to_restore if "value_output" not in v.name}

    # vars_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent0')
    # for v in vars_train:
    #     print(v.name + ':', tf.shape(v))
    saver = tf.train.Saver(variables_to_restore, max_to_keep=100)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # variables_names = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent0')]
        # sess.run(tf.global_variables_initializer())
        # values = sess.run(variables_names)
        # for k, v in zip(variables_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./Model/accuracy_fake_minor/model-9500")
        master.update_params_from_agent0(sess)
        master.run(sess, saver, 300)
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

