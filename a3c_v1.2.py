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
    def __init__(self, name, trainer):
        self.name = name
        self.episodes = tf.Variable(0, dtype=tf.int32, name='episodes_' + name, trainable=False)
        self.increment = self.episodes.assign_add(1)
        self.main_network = CardNetwork(54 * 6, trainer, self.name)
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
        active_response_input, advantages, val_truth, train_decision, train_response = [buffer[i] for i in range(2, 22)]

        # pick up masks
        passive_decision_mask_input, passive_response_mask_input, passive_bomb_mask_input, \
        active_decision_mask_input, active_response_mask_input, active_seq_length_mask_input = [buffer[i] for i in
                                                                                                range(22, 28)]

        # main network training
        decision_passive_output, response_passive_output, bomb_passive_output, \
        decision_active_output, response_active_output, main_loss, main_val_loss, \
        active_decision_loss, active_response_loss, passive_decision_loss, passive_response_loss, passive_bomb_loss, main_grads, _ \
            = sess.run([self.main_network.fc_decision_passive_output,
                        self.main_network.fc_response_passive_output, self.main_network.fc_bomb_passive_output,
                        self.main_network.fc_decision_active_output, self.main_network.fc_response_active_output,
                        self.main_network.loss, self.main_network.val_loss,
                        self.main_network.active_decision_loss, self.main_network.active_response_loss,
                        self.main_network.passive_decision_loss,
                        self.main_network.passive_response_loss, self.main_network.passive_bomb_loss,
                        self.main_network.gradients,
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
                           self.main_network.active_response_input: np.array([active_response_input]),
                           self.main_network.train_decision: np.array([train_decision]),
                           self.main_network.train_response: np.array([train_response]),
                           self.main_network.passive_decision_mask: passive_decision_mask_input.reshape(1, -1).astype(
                               bool),
                           self.main_network.passive_response_mask: passive_response_mask_input.reshape(1, -1).astype(
                               bool),
                           self.main_network.passive_bomb_mask: passive_bomb_mask_input.reshape(1, -1).astype(bool),
                           self.main_network.active_decision_mask: active_decision_mask_input.reshape(1, -1).astype(
                               bool),
                           self.main_network.active_response_mask: active_response_mask_input.reshape(1, -1).astype(
                               bool),
                           self.main_network.seq_length_mask: active_seq_length_mask_input.reshape(1, -1).astype(bool)
                       })

        episode = sess.run(self.episodes)
        return [decision_passive_output, response_passive_output, bomb_passive_output,
                decision_active_output, response_active_output, main_loss, main_val_loss,
                active_decision_loss, active_response_loss, passive_decision_loss, passive_response_loss,
                passive_bomb_loss,
                main_grads]

    def train_batch_sampled(self, buffer, batch_size):
        batch_idx = np.random.choice(len(buffer), batch_size)
        input_states, input_singles, input_pairs, input_triples, input_quadrics, vals, modes = [np.array([buffer[i][j] for i in batch_idx]) for j in range(7)]
        passive_decision_input, passive_response_input, passive_bomb_input, active_decision_input, \
            active_response_input, seq_length_input = [np.array([buffer[i][7][k] for i in batch_idx])
                                                       for k in ['decision_passive', 'bomb_passive', 'response_passive',
                                                                 'decision_active', 'response_active', 'seq_length']]
        _, loss, gradients, decision_passive_output, response_passive_output, bomb_passive_output, \
            decision_active_output, response_active_output, seq_length_output \
            = sess.run([self.main_network.optimize, self.main_network.loss,
                        self.main_network.gradients, self.main_network.fc_decision_passive_output,
                        self.main_network.fc_response_passive_output, self.main_network.fc_bomb_passive_output,
                        self.main_network.fc_decision_active_output, self.main_network.fc_response_active_output,
                        self.main_network.fc_sequence_length_output
                        ],
                       feed_dict={
                           self.main_network.training: True,
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
                           self.main_network.value_input: vals
                       })
        return loss, gradients


class CardMaster:
    def __init__(self):
        self.temp = 1
        self.start_temp = 1
        self.end_temp = 0.2
        self.name = 'global'
        self.env = Pyenv()
        self.sess = None

        self.train_intervals = 10

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.episode_rewards = [[] for i in range(3)]
        self.episode_length = [[] for i in range(3)]
        self.episode_mean_values = [[] for i in range(3)]
        self.summary_writers = [tf.summary.FileWriter("train_agent%d" % i) for i in range(3)]

        self.agents = [CardAgent('agent%d' % i, self.trainer) for i in range(3)]

        self.global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        self.increment = self.global_episodes.assign_add(1)

    def train_batch(self, buffer, sess, gamma, val_last, idx):
        buffer = np.array(buffer)
        return self.agents[idx].train_batch(buffer, sess, gamma, val_last)

    def train_batch_packed(self, buffer, sess, gamma, val_last, idx):
        return self.agents[idx].train_batch_packed(buffer, sess, gamma, val_last)

    def train_batch_sampled(self, buffer, batch_size):
        for i in range(len(buffer)):
            self.agents[i].train_batch_sampled(buffer[i], batch_size)

    def respond(self, env):
        mask = get_mask(to_char(self.env.get_curr_handcards()), self.action_space,
                        to_char(self.env.get_last_outcards()))
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

                for l in range(max_episode_length):
                    s = self.env.get_state()
                    # time.sleep(1)
                    # # map 1-3 to 0-2
                    train_id = self.env.get_role_ID() - 1

                    curr_cards_char = self.env.get_handcards()
                    last_cards_char = self.env.get_last_outcards()

                    input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, last_cards_char)

                    dump_s = self.env.dump_state()
                    mctree = MCTree(dump_s)
                    mctree.search(1, 100)

                    mode, distribution, intention = mctree.step(1.)
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

                    if done:
                        for buf in episode_buffer[train_id]:
                            # update reward, with no decay
                            buf[-1] = r
                        for i in range(3):
                            if i == train_id:
                                continue
                            if train_id == lord_idx:
                                for buf in episode_buffer[i]:
                                    # update reward, with no decay
                                    buf[-1] = -r
                            elif i == lord_idx:
                                for buf in episode_buffer[i]:
                                    # update reward, with no decay
                                    buf[-1] = -r
                            else:
                                for buf in episode_buffer[i]:
                                    # update reward, with no decay
                                    buf[-1] = r

                        # then sample buffer for training
                        self.train_batch_sampled([episode_buffer[i] for i in range(3)], 8)

                    # episode_values[train_id].append(val_output[0])
                    episode_reward[train_id] += r
                    episode_steps[train_id] += 1

                for i in range(3):
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
                        # TODO: add more summary
                        summary.value.add(tag='Performance/rewards', simple_value=float(mean_reward))
                        summary.value.add(tag='Performance/length', simple_value=float(mean_length))
                        summary.value.add(tag='Performance/values', simple_value=float(mean_value))
                        # summary.value.add(tag='Losses/Value Loss', simple_value=float(val_loss))
                        # summary.value.add(tag='Losses/Prob pred', simple_value=float(pred_prob))
                        # summary.value.add(tag='Losses/Policy Loss', simple_value=float(policy_loss))
                        # summary.value.add(tag='Losses/Grad Norm', simple_value=float(grad_norms))
                        # summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
                        # summary.value.add(tag='Losses/Policy Norm', simple_value=float(p_norm))
                        # summary.value.add(tag='Losses/a0', simple_value=float(a0))
                        self.summary_writers[i].add_summary(summary, episodes)
                        self.summary_writers[i].flush()

                global_episodes += 1
                sess.run(self.increment)
                if global_episodes % 1000 == 0:
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
        sess.close()