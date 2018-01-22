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
from pyenv import Pyenv
from utils import get_masks, discard_cards
from montecarlo import MCTree
from network_RL import CardNetwork


class CardAgent:
    def __init__(self, name):
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.name = name
        self.episodes = tf.Variable(0, dtype=tf.int32, name='episodes_' + name, trainable=False)
        self.increment = self.episodes.assign_add(1)
        self.main_network = CardNetwork(54 * 6, self.trainer, self.name)

    def evaluate(self, s, sess):
        assert not s.is_intermediate()
        curr_hands_char = s['player_cards'][s['idx']]
        input_single, input_pair, input_triple, input_quadric = get_masks(curr_hands_char,
                                                                          s['last_cards'] if s['idx'] != s[
                                                                              'control_idx'] else None)
        val = sess.run(self.main_network.fc_value_output,
                       feed_dict={
                           self.main_network.training: True,
                           self.main_network.input_state: Pyenv.get_state_static(s).reshape(1, -1),
                           self.main_network.input_single: input_single.reshape(1, -1),
                           self.main_network.input_pair: input_pair.reshape(1, -1),
                           self.main_network.input_triple: input_triple.reshape(1, -1),
                           self.main_network.input_quadric: input_quadric.reshape(1, -1)
                       })[0]
        return val

    def predict(self, s, valid_space, sess):
        stage = s['stage']
        curr_hands_char = s['player_cards'][s['idx']]
        input_single, input_pair, input_triple, input_quadric = get_masks(curr_hands_char, s['last_cards'] if s['idx'] != s['control_idx'] else None)
        feeddict = {
            self.main_network.training: True,
            self.main_network.input_state: Pyenv.get_state_static(s).reshape(1, -1),
            self.main_network.input_single: input_single.reshape(1, -1),
            self.main_network.input_pair: input_pair.reshape(1, -1),
            self.main_network.input_triple: input_triple.reshape(1, -1),
            self.main_network.input_quadric: input_quadric.reshape(1, -1)
        }
        if stage == 'p_decision':
            decision_output = sess.run(self.main_network.fc_decision_passive_output,
                                       feed_dict=feeddict)[0]
            return decision_output[valid_space]
        elif stage == 'p_bomb':
            bomb_output = sess.run(self.main_network.fc_bomb_passive_output,
                                   feed_dict=feeddict)[0]
            return bomb_output[valid_space]
        elif stage == 'p_response':
            response_output = sess.run(self.main_network.fc_response_passive_output,
                                       feed_dict=feeddict)[0]
            return response_output[valid_space]
        elif stage == 'a_decision':
            decision_output = sess.run(self.main_network.fc_decision_active_output,
                                       feed_dict=feeddict)[0]
            return decision_output[valid_space]
        elif stage == 'a_response':
            response_output = sess.run(self.main_network.fc_response_active_output,
                                       feed_dict=feeddict)[0]
            return response_output[valid_space]
        elif stage == 'a_length':
            length_output = sess.run(self.main_network.fc_sequence_length_output,
                                       feed_dict=feeddict)[0]
            return length_output[valid_space]
        elif stage == 'minor':
            if 'minor_cards' in s:
                card_history = s['main_cards'] + s['minor_cards']
            else:
                card_history = s['main_cards']
            curr_hands_char = discard_cards(curr_hands_char, card_history)
            input_single, input_pair, input_triple, input_quadric = get_masks(curr_hands_char,
                                                                              s['last_cards'] if s['idx'] != s[
                                                                                  'control_idx'] else None)
            # TODO: correct for state
            state = Pyenv.get_state_static(s)
            cards_onehot = card.Card.char2onehot(card_history)

            state[:54] -= cards_onehot
            state[2 * 54:3 * 54] += cards_onehot
            minor_output = sess.run(self.main_network.fc_response_active_output,
                                    feed_dict={
                                        self.main_network.training: True,
                                        self.main_network.input_state: state.reshape(1, -1),
                                        self.main_network.input_single: input_single.reshape(1, -1),
                                        self.main_network.input_pair: input_pair.reshape(1, -1),
                                        self.main_network.input_triple: input_triple.reshape(1, -1),
                                        self.main_network.input_quadric: input_quadric.reshape(1, -1)
            })[0]
            return minor_output[valid_space]
        else:
            raise Exception('unexpected stage name')

    def train_batch_sampled(self, buf, batch_size, sess):
        num_of_iters = len(buf) // batch_size + 1
        mean_loss = 0
        mean_var_norm = 0
        mean_grad_norm = 0
        for _ in range(num_of_iters):
            batch_idx = np.random.choice(len(buf), batch_size)
            input_states, input_singles, input_pairs, input_triples, input_quadrics, vals, modes = [np.array([buf[i][j] for i in batch_idx]) for j in range(7)]
            passive_decision_input, passive_bomb_input, passive_response_input, active_decision_input, \
                active_response_input, seq_length_input = [np.array([buf[i][7][k] for i in batch_idx])
                                                           for k in ['decision_passive', 'bomb_passive', 'response_passive',
                                                                     'decision_active', 'response_active', 'seq_length']]
            _, loss, var_norm, gradient_norm, decision_passive_output, response_passive_output, bomb_passive_output, \
                decision_active_output, response_active_output, seq_length_output \
                = sess.run([self.main_network.optimize, self.main_network.loss, self.main_network.var_norms,
                            self.main_network.grad_norms, self.main_network.fc_decision_passive_output,
                            self.main_network.fc_response_passive_output, self.main_network.fc_bomb_passive_output,
                            self.main_network.fc_decision_active_output, self.main_network.fc_response_active_output,
                            self.main_network.fc_sequence_length_output
                            ],
                           feed_dict={
                               self.main_network.training: True,
                               self.main_network.mode: modes,
                               self.main_network.input_state: input_states,
                               self.main_network.input_single: input_singles,
                               self.main_network.input_pair: input_pairs,
                               self.main_network.input_triple: input_triples,
                               self.main_network.input_quadric: input_quadrics,
                               self.main_network.passive_decision_input: passive_decision_input,
                               self.main_network.passive_response_input: passive_response_input,
                               self.main_network.passive_bomb_input: passive_bomb_input,
                               self.main_network.active_decision_input: active_decision_input,
                               self.main_network.active_response_input: active_response_input,
                               self.main_network.seq_length_input: seq_length_input,
                               self.main_network.value_input: vals
                           })
            mean_loss += loss
            mean_grad_norm += gradient_norm
            mean_var_norm += var_norm
        mean_loss /= num_of_iters
        mean_var_norm /= num_of_iters
        mean_grad_norm /= num_of_iters
        return mean_loss, mean_var_norm, mean_grad_norm


class CardMaster:
    def __init__(self):
        self.temp = 1
        self.start_temp = 1
        self.end_temp = 0.2
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

    def train_batch_sampled(self, buf, batch_size, sess):
        logs = []
        for i in range(len(buf)):
            logs.append(self.agents[i].train_batch_sampled(buf[i], batch_size, sess))
        return logs

    # train three agents simultaneously
    def run(self, sess, saver, max_episode_length):
        self.sess = sess
        with sess.as_default():
            global_episodes = sess.run(self.global_episodes)
            total_episodes = 10001
            temp_decay = (self.end_temp - self.start_temp) / total_episodes
            while global_episodes < total_episodes:
                self.env.reset()
                self.env.prepare()
                lord_idx = self.env.lord_idx
                print("episode %d" % global_episodes)
                episode_buffer = [[] for _ in range(3)]
                episode_values = [[] for _ in range(3)]
                episode_reward = [0, 0, 0]
                episode_steps = [0, 0, 0]

                logs = []
                for l in range(max_episode_length):
                    s = self.env.get_state()
                    # time.sleep(1)
                    # # map 1-3 to 0-2
                    train_id = self.env.get_role_ID() - 1

                    curr_cards_char = self.env.get_handcards()
                    last_cards_char = self.env.get_last_outcards()

                    input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, last_cards_char)

                    dump_s = self.env.dump_state()
                    mctree = MCTree(dump_s, self.agents[train_id], self.sess)
                    mctree.search(1, 10)
                    # print(train_id, 'single step')

                    # TODO: decay temperature through time
                    temp = self.start_temp + global_episodes * temp_decay
                    mode, distribution, intention = mctree.step(temp)
                    print('lord: ' if train_id == 1 else 'farmer: ', intention)
                    has_minor = False
                    if mode > 4:
                        mode = mode - 5
                        has_minor = True
                    episode_buffer[train_id].append([s.copy(), input_single, input_pair, input_triple, input_quadric,
                                                     0, mode, distribution])

                    if has_minor:
                        for i in range(len(distribution['minor_cards'])):
                            cards = distribution['cards_history'][i]
                            cards_onehot = card.Card.char2onehot(cards)

                            s[:54] -= cards_onehot
                            s[2 * 54:3 * 54] += cards_onehot
                            discard_cards(curr_cards_char, cards)

                            input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, None)
                            distribution_copy = distribution.copy()
                            distribution_copy['response_active'] = distribution['minor_cards'][i]

                            episode_buffer[train_id].append([s.copy(), input_single, input_pair, input_triple,
                                                             input_quadric, 0, 5, distribution_copy])

                    r, done = self.env.step(intention)

                    episode_reward[train_id] += r
                    episode_steps[train_id] += 1
                    # self.train_batch_sampled([episode_buffer[train_id]], 8)

                    if done:
                        for buf in episode_buffer[train_id]:
                            # update reward, with no decay
                            buf[-3] = r

                        for i in range(3):
                            if i == train_id:
                                continue
                            if train_id == lord_idx:
                                for buf in episode_buffer[i]:
                                    # update reward, with no decay
                                    buf[-3] = -r
                                    episode_reward[i] += -r
                            elif i == lord_idx:
                                for buf in episode_buffer[i]:
                                    # update reward, with no decay
                                    buf[-3] = -r
                                    episode_reward[i] += -r
                            else:
                                for buf in episode_buffer[i]:
                                    # update reward, with no decay
                                    buf[-3] = r
                                    episode_reward[i] += r

                        # then sample buffer for training
                        logs = self.train_batch_sampled([episode_buffer[i] for i in range(3)], 8, sess)
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
    master = CardMaster()
    saver = tf.train.Saver(max_to_keep=100)
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