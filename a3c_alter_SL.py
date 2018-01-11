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
from logger import Logger
from network_SL import CardNetwork
from utils import get_mask, get_minor_cards, train_fake_action, get_masks, test_fake_action
from utils import get_seq_length, pick_minor_targets, to_char, to_value, get_mask_alter
import shutil


##################################################### UTILITIES ########################################################


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
        self.action_space = action_space
        self.name = 'global'
        self.env = env
        self.a_dim = 9085
        self.gamma = 0.99
        self.sess = None

        self.train_intervals = 30

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
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
                    
                    mask = get_mask(to_char(self.env.get_curr_handcards()), self.action_space, to_char(self.env.get_last_outcards()))

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
    parser.add_argument('--epoches_train', type=int, help='num of epochs to train', default=10000)
    parser.add_argument('--epoches_test', type=int, help='num of epochs to test', default=1000)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.set_defaults(train=True)

    args = parser.parse_args(sys.argv[1:])
    epoches_train = args.epoches_train
    epoches_test = args.epoches_test

    a_dim = len(action_space)

    graph_sl = tf.get_default_graph()
    SLNetwork = CardNetwork(54 * 6, tf.train.AdamOptimizer(learning_rate=0.0001), "SLNetwork")
    variables = tf.all_variables()

    e = env.Env()
    TRAIN = args.train
    sess = tf.Session(graph=graph_sl)
    saver = tf.train.Saver(max_to_keep=50)

    file_writer = tf.summary.FileWriter('accuracy_fake_minor', sess.graph)

    # filelist = [ f for f in os.listdir('./accuracy_fake_minor') ]
    # for f in filelist:
    #     os.remove(os.path.join('./accuracy_fake_minor', f))

    logger = Logger()
    # TODO: support batch training
    # test_cards = [i for i in range(3, 18)]
    # test_cards = np.array(test_cards *)
    if TRAIN:
        sess.run(tf.global_variables_initializer())
        for i in range(epoches_train):
            e.reset()
            e.prepare()
            
            curr_cards_value = e.get_curr_handcards()
            curr_cards_char = to_char(curr_cards_value)
            last_cards_value = e.get_last_outcards()
            # print("curr_cards = ", curr_cards_value)
            # print("last_cards = ", last_cards_value)
            last_category_idx = -1
            # last_cards_char = to_char(last_cards_value)
            # mask = get_mask(curr_cards_char, action_space, last_cards_char)

            input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, None)

            s = e.get_state()
            s = np.reshape(s, [1, -1])
            
            # s = get_feature_state(e, mask)
            r = 0
            while r == 0:
                active = True
                if last_cards_value.size > 0:
                    active = False

                has_seq_length = np.array([False])
                seq_length_input = np.array([0])
                is_passive_bomb = np.array([False])
                did_passive_response = np.array([False])
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
                # do not account for 4 + 2 + 2 
                if category_idx == 14:
                    curr_cards_value = e.get_curr_handcards()
                    curr_cards_char = to_char(curr_cards_value)
                    last_cards_value = e.get_last_outcards()
                    if last_cards_value.size > 0:
                        last_category_idx = category_idx
                    # last_cards_char = to_char(last_cards_value)
                    # mask = get_mask(curr_cards_char, action_space, last_cards_char)

                    input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, None)
                    continue

                # if category_idx == Category.THREE_ONE.value or category_idx == Category.THREE_TWO.value or \
                #         category_idx == Category.THREE_ONE_LINE.value or category_idx == Category.THREE_TWO_LINE.value or \
                #         category_idx == Category.FOUR_TWO.value:
                #     has_minor_cards[0] = True
                #     minor_cards_target[0], minor_cards_length[0] = get_minor_cards(intention, category_idx)
                minor_cards_targets = pick_minor_targets(category_idx, to_char(intention))
                
                if not active:
                    if category_idx == Category.QUADRIC.value and category_idx != last_category_idx:
                        is_passive_bomb[0] = True
                        passive_decision_input[0] = 1
                        passive_bomb_input[0] = intention[0] - 3
                    else:
                        if category_idx == Category.BIGBANG.value:
                            passive_decision_input[0] = 2
                        else:
                            if category_idx != Category.EMPTY.value:
                                passive_decision_input[0] = 3
                                did_passive_response[0] = True
                                # OFFSET_ONE
                                passive_response_input[0] = intention[0] - last_cards_value[0] - 1
                                if passive_response_input[0] < 0:
                                    print("something bad happens")
                                    passive_response_input[0] = 0
                else:
                    seq_length = get_seq_length(category_idx, intention)
                    if seq_length is not None:
                        has_seq_length[0] = True
                        # length offset one
                        seq_length_input[0] = seq_length - 1

                    # ACTIVE OFFSET ONE!
                    active_decision_input[0] = category_idx - 1
                    active_response_input[0] = intention[0] - 3
                    

                _, decision_passive_output, response_passive_output, bomb_passive_output, \
                    decision_active_output, response_active_output, seq_length_output, minor_cards_output, loss, \
                    active_decision_loss, active_response_loss, passive_decision_loss, passive_response_loss, passive_bomb_loss \
                     = sess.run([SLNetwork.optimize, SLNetwork.fc_decision_passive_output, 
                                SLNetwork.fc_response_passive_output, SLNetwork.fc_bomb_passive_output,
                                SLNetwork.fc_decision_active_output, SLNetwork.fc_response_active_output, 
                                SLNetwork.fc_sequence_length_output,
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
                            SLNetwork.is_passive_bomb: is_passive_bomb,
                            SLNetwork.did_passive_response: did_passive_response,
                            SLNetwork.passive_decision_input: passive_decision_input,
                            SLNetwork.passive_response_input: passive_response_input,
                            SLNetwork.passive_bomb_input: passive_bomb_input,
                            SLNetwork.active_decision_input: active_decision_input,
                            SLNetwork.active_response_input: active_response_input,
                            SLNetwork.has_seq_length: has_seq_length,
                            SLNetwork.seq_length_input: seq_length_input
                })

                if minor_cards_targets is not None:
                    accs = train_fake_action(minor_cards_targets, curr_cards_char.copy(), s, sess, SLNetwork, category_idx)
                    for acc in accs:
                        logger.updateAcc("minor_cards", acc)

                #print("gradients ： ", gradients)

                # update accuracies
                # if has_minor_cards[0]:
                #     # print("minor_cards_output : ", minor_cards_output)
                #     # print("minor_cards_output.argsort : ", minor_cards_output.argsort()[:int(minor_cards_length[0])])
                #     # print("minor_cards_target[0] : ", minor_cards_target[0])
                #     # print("minor_cards_target[0].argsort : ", minor_cards_target[0].argsort()[:int(minor_cards_length[0])])
                #     minor_cards_acc_temp = 1 if np.array_equal(minor_cards_output.argsort()[:int(minor_cards_length[0])][0], \
                #         minor_cards_target.argsort()[:int(minor_cards_length[0])][0]) else 0
                #     logger.updateAcc("minor_cards", minor_cards_acc_temp)
                if active:
                    active_decision_acc_temp = 1 if np.argmax(decision_active_output) == active_decision_input[0] else 0
                    logger.updateAcc("active_decision", active_decision_acc_temp)

                    active_response_acc_temp = 1 if np.argmax(response_active_output) == active_response_input[0] else 0
                    logger.updateAcc("active_response", active_response_acc_temp)

                    if has_seq_length[0]:
                        seq_acc_temp = 1 if np.argmax(seq_length_output) == seq_length_input[0] else 0
                        logger.updateAcc("seq_length", seq_acc_temp)
                else:
                    passive_decision_acc_temp = 1 if np.argmax(decision_passive_output) == passive_decision_input[0] else 0
                    logger.updateAcc("passive_decision", passive_decision_acc_temp)

                    if is_passive_bomb[0]:
                        passive_bomb_acc_temp = 1 if np.argmax(bomb_passive_output) == passive_bomb_input[0] else 0
                        logger.updateAcc("passive_bomb", passive_bomb_acc_temp)
                    elif passive_decision_input[0] == 3:
                        passive_response_acc_temp = 1 if np.argmax(response_passive_output) == passive_response_input[0] else 0
                        logger.updateAcc("passive_response", passive_response_acc_temp)

                curr_cards_value = e.get_curr_handcards()
                curr_cards_char = to_char(curr_cards_value)
                last_cards_value = e.get_last_outcards()
                # print("curr_cards = ", curr_cards_value)
                # print("last_cards = ", last_cards_value)
                if last_cards_value.size > 0:
                    last_category_idx = category_idx
                    last_cards_char = to_char(last_cards_value)
                # mask = get_mask(curr_cards_char, action_space, last_cards_char)

                input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, last_cards_char if last_cards_value.size > 0 else None)

                s = e.get_state()
                s = np.reshape(s, [1, -1])
            #print("End of one game")
            if i % 100 == 0:
                print("train1 ", i, " ing...")
                print("train passive decision accuracy = ", logger["passive_decision"])
                print("train passive response accuracy = ", logger["passive_response"])
                print("train passive bomb accuracy = ", logger["passive_bomb"])
                print("train active decision accuracy = ", logger["active_decision"])
                print("train active response accuracy = ", logger["active_response"])
                print("train sequence length accuracy = ", logger["seq_length"])
                print("train minor cards accuracy = ", logger["minor_cards"])
                summary = tf.Summary(value=[
                                tf.Summary.Value(tag="Accuracy/passive_decision_accuracy", simple_value=logger["passive_decision"]), 
                                tf.Summary.Value(tag="Accuracy/passive_response_accuracy", simple_value=logger["passive_response"]),
                                tf.Summary.Value(tag="Accuracy/passive_bomb_accuracy", simple_value=logger["passive_bomb"]),
                                tf.Summary.Value(tag="Accuracy/active_decision_accuracy", simple_value=logger["active_decision"]),
                                tf.Summary.Value(tag="Accuracy/active_response_accuracy", simple_value=logger["active_response"]),
                                tf.Summary.Value(tag="Accuracy/seq_length_accuracy", simple_value=logger["seq_length"]),
                                tf.Summary.Value(tag="Accuracy/minor_cards_accuracy", simple_value=logger["minor_cards"]),
                                #tf.Summary.Value(tag="minor_loss", simple_value=minor_loss)
                                tf.Summary.Value(tag="Loss/loss", simple_value=loss[0]),
                                tf.Summary.Value(tag="Loss/active_decision_loss", simple_value=active_decision_loss[0]),
                                tf.Summary.Value(tag="Loss/active_response_loss", simple_value=active_response_loss[0]),
                                tf.Summary.Value(tag="Loss/passive_decision_loss", simple_value=passive_decision_loss[0]),
                                tf.Summary.Value(tag="Loss/passive_response_loss", simple_value=passive_response_loss[0]),
                                tf.Summary.Value(tag="Loss/passive_bomb_loss", simple_value=passive_bomb_loss[0]) 
                                #tf.Summary.Value(tag="gradients", )
                ])
                file_writer.add_summary(summary, i / 100 - 1)
            if i % 200 == 0:
                saver.save(sess, "./Model/accuracy_fake_minor/model", global_step=i)

        
        # saver.save(sess, "./Model/SLNetwork_feat_deeper_1000000epoches.ckpt")

        print("train passive decision accuracy = ", logger["passive_decision"])
        print("train passive response accuracy = ", logger["passive_response"])
        print("train passive bomb accuracy = ", logger["passive_bomb"])
        print("train active decision accuracy = ", logger["active_decision"])
        print("train active response accuracy = ", logger["active_response"])
        print("train seq length accuracy = ", logger["seq_length"])
        print("train minor cards accuracy = ", logger["minor_cards"])

        file_writer.close()

    # RL part #################################################################
    # print('loading model...')
    # saver.restore(sess, "./Model/accuracy_bugfixed_lr0001_1110/model.ckpt")

    # graph_rl = tf.Graph()
    # with graph_rl.as_default():
    #     sess = tf.Session(graph = graph_rl)
    #     cardgame = env.Env()
    #     master = CardMaster(cardgame)
    #     master.run(sess, saver, 2000)
    # RL part #################################################################

    # test part
    # saver.restore(sess, "./Model/SLNetwork_feat_deeper_1000000epoches.ckpt")

    if not TRAIN:
        saver.restore(sess, "./Model/accuracy_fake_minor/model-9800")
        for i in range(epoches_test):
            e.reset()
            e.prepare()
        
            r = 0
            done = False
            while r == 0:
                curr_cards_value = e.get_curr_handcards()
                curr_cards_char = to_char(curr_cards_value)
                last_cards_value = e.get_last_outcards()
                last_cards_char = to_char(last_cards_value)
                # print("curr_cards = ", curr_cards_char)
                # print("last_cards = ", last_cards_value)
                last_category_idx = e.get_last_outcategory_idx()

                input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, last_cards_char if last_cards_value.size > 0 else None)

                s = e.get_state()
                s = np.reshape(s, [1, -1])

                intention, r, category_idx = e.step_auto()       

                is_passive_bomb = False
                has_seq_length = False
                is_active = (last_cards_value.size == 0)
                if is_active:
                    # first get mask
                    decision_mask, response_mask, _, length_mask = get_mask_alter(curr_cards_char, [], False, last_category_idx)

                    decision_active_output = sess.run(SLNetwork.fc_decision_active_output,
                        feed_dict={
                            SLNetwork.training: False,
                            SLNetwork.input_state: s,
                            SLNetwork.input_single: np.reshape(input_single, [1, -1]),
                            SLNetwork.input_pair: np.reshape(input_pair, [1, -1]),
                            SLNetwork.input_triple: np.reshape(input_triple, [1, -1]),
                            SLNetwork.input_quadric: np.reshape(input_quadric, [1, -1])
                        })

                    # make decision depending on output
                    decision_active_output = decision_active_output[0]
                    decision_active_output[decision_mask == 0] = -1
                    decision_active_pred = np.argmax(decision_active_output)
                    decision_active_target = category_idx - 1
                    
                    active_category_idx = decision_active_target + 1

                    # give actual response
                    response_active_output = sess.run(SLNetwork.fc_response_active_output,
                        feed_dict={
                            SLNetwork.training: False,
                            SLNetwork.input_state: s,
                            SLNetwork.input_single: np.reshape(input_single, [1, -1]),
                            SLNetwork.input_pair: np.reshape(input_pair, [1, -1]),
                            SLNetwork.input_triple: np.reshape(input_triple, [1, -1]),
                            SLNetwork.input_quadric: np.reshape(input_quadric, [1, -1])
                        })

                    response_active_output = response_active_output[0]
                    response_active_output[response_mask[decision_active_target] == 0] = -1
                    response_active_pred = np.argmax(response_active_output)
                    response_active_target = intention[0] - 3

                    seq_length = get_seq_length(category_idx, intention)
                    if seq_length is not None:
                        has_seq_length = True
                        seq_length_target = seq_length

                    seq_length = 0

                    # next sequence length
                    if active_category_idx == Category.SINGLE_LINE.value or \
                            active_category_idx == Category.DOUBLE_LINE.value or \
                            active_category_idx == Category.TRIPLE_LINE.value or \
                            active_category_idx == Category.THREE_ONE_LINE.value or \
                            active_category_idx == Category.THREE_TWO_LINE.value:
                        seq_length_output = sess.run(SLNetwork.fc_sequence_length_output,
                            feed_dict={
                                SLNetwork.training: False,
                                SLNetwork.input_state: s,
                                SLNetwork.input_single: np.reshape(input_single, [1, -1]),
                                SLNetwork.input_pair: np.reshape(input_pair, [1, -1]),
                                SLNetwork.input_triple: np.reshape(input_triple, [1, -1]),
                                SLNetwork.input_quadric: np.reshape(input_quadric, [1, -1])
                            })

                        seq_length_output = seq_length_output[0]
                        seq_length_output[length_mask[decision_active_target][response_active_target] == 0] = -1
                        seq_length_pred = np.argmax(seq_length_output) + 1
                    
                else:
                    is_bomb = False
                    if len(last_cards_value) == 4 and len(set(last_cards_value)) == 1:
                        is_bomb = True
                    decision_mask, response_mask, bomb_mask, _ = get_mask_alter(curr_cards_char, to_char(last_cards_value), is_bomb, last_category_idx)

                    decision_passive_output, response_passive_output, bomb_passive_output \
                        = sess.run([SLNetwork.fc_decision_passive_output,
                                    SLNetwork.fc_response_passive_output, SLNetwork.fc_bomb_passive_output],
                                    feed_dict={
                                        SLNetwork.training: False,
                                        SLNetwork.input_state: s,
                                        SLNetwork.input_single: np.reshape(input_single, [1, -1]),
                                        SLNetwork.input_pair: np.reshape(input_pair, [1, -1]),
                                        SLNetwork.input_triple: np.reshape(input_triple, [1, -1]),
                                        SLNetwork.input_quadric: np.reshape(input_quadric, [1, -1])
                                    })
                    
                    decision_passive_target = 0
                    if category_idx == Category.QUADRIC.value and category_idx != last_category_idx:
                        is_passive_bomb = True
                        decision_passive_target = 1
                        bomb_passive_target = intention[0] - 3
                    else:
                        if category_idx == Category.BIGBANG.value:
                            decision_passive_target = 2
                        else:
                            if category_idx != Category.EMPTY.value:
                                decision_passive_target = 3
                                # OFFSET_ONE
                                response_passive_target = intention[0] - last_cards_value[0] - 1
                    
                    decision_passive_output = decision_passive_output[0]
                    decision_passive_output[decision_mask == 0] = -1
                    decision_passive_pred = np.argmax(decision_passive_output)
                    

                    if decision_passive_target == 0:
                        pass
                    elif decision_passive_target == 1:
                        # print('bomb_mask', bomb_mask)
                        bomb_passive_output = bomb_passive_output[0]
                        bomb_passive_output[bomb_mask == 0] = -1
                        bomb_passive_pred = np.argmax(bomb_passive_output)
                    elif decision_passive_target == 2:
                        pass
                    elif decision_passive_target == 3:
                        # print('response_mask', response_mask)
                        response_passive_output = response_passive_output[0]
                        response_passive_output[response_mask == 0] = -1
                        response_passive_pred = np.argmax(response_passive_output)

                minor_cards_targets = pick_minor_targets(category_idx, to_char(intention))
                if minor_cards_targets is not None:
                    # print(minor_cards_targets)
                    # print(curr_cards_char)
                    dup_mask = np.ones([15])
                    dup_mask[intention[0] - 3] = 0
                    accs = test_fake_action(minor_cards_targets, curr_cards_char.copy(), s, sess, SLNetwork, category_idx, dup_mask)
                    for acc in accs:
                        logger.updateAcc("minor_cards", acc)

                if is_active:
                    active_decision_acc_temp = 1 if decision_active_pred == decision_active_target else 0
                    logger.updateAcc("active_decision", active_decision_acc_temp)

                    active_response_acc_temp = 1 if response_active_pred == response_active_target else 0
                    logger.updateAcc("active_response", active_response_acc_temp)

                    if has_seq_length:
                        seq_acc_temp = 1 if seq_length_pred == seq_length_target else 0
                        logger.updateAcc("seq_length", seq_acc_temp)
                else:
                    passive_decision_acc_temp = 1 if decision_passive_pred == decision_passive_target else 0
                    logger.updateAcc("passive_decision", passive_decision_acc_temp)

                    if is_passive_bomb:
                        passive_bomb_acc_temp = 1 if bomb_passive_pred == bomb_passive_target else 0
                        logger.updateAcc("passive_bomb", passive_bomb_acc_temp)
                    elif decision_passive_target == 3:
                        passive_response_acc_temp = 1 if response_passive_pred == response_passive_target else 0
                        logger.updateAcc("passive_response", passive_response_acc_temp)
            
            if i % 100 == 0:
                print("test ", i, " ing...")
                print("test passive decision accuracy = ", logger["passive_decision"])
                print("test passive response accuracy = ", logger["passive_response"])
                print("test passive bomb accuracy = ", logger["passive_bomb"])
                print("test active decision accuracy = ", logger["active_decision"])
                print("test active response accuracy = ", logger["active_response"])
                print("test sequence length accuracy = ", logger["seq_length"])
                print("test minor cards accuracy = ", logger["minor_cards"])

                
