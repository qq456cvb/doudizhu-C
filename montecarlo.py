import numpy as np
import math
from pyenv import Pyenv
import card
from card import Category
import threading
import multiprocessing
from utils import to_char, to_value, give_cards_without_minor
import copy
from time import sleep

import sys
sys.path.insert(0, './build/Release')
import env


class Environment:
    def __init__(self):
        self.board = np.zeros([3, 3])

    def __str__(self):
        return '\n'.join([' '.join([str(int(self.board[i,j])) for i in range(3)]) for j in range(3)])

    def step(self, s, a, pid):
        s[a//3, a%3] = pid
        if self.is_over(s) > 0:
            return 1, True, s, 3 - pid
        return 0, False, s, 3 - pid

    def get_actions(self, s):
        actions = []
        for i in range(3):
            for j in range(3):
                if s[i,j] == 0:
                    actions.append(i * 3 + j)
        return actions

    def stringify(self, s):
        return '\n'.join([' '.join([str(int(s[i,j])) for j in range(3)]) for i in range(3)])

    def is_over(self, s):
        for i in range(3):
            for j in range(1, 3):
                if np.all(s[i,:] == j) or np.all(s[:,i] == j):
                    return j
        for j in range(1, 3):
            if np.all(np.array([s[i,i] for i in range(3)]) == j) or np.all(np.array([s[2-i,i] for i in range(3)]) == j):
                return j
        over = True
        for i in range(3):
            for j in range(3):
                if s[i,j] == 0:
                    over = False
                    break
        return -1 if over else 0


class Evaluator:
    def __init__(self, env):
        self.env = env

    def eval_helper(self, s, pid, depth):
        over = self.env.is_over(s)
        if over > 0:
            return (1, 1)
        if over == -1:
            return (0, 0)
        actions = self.env.get_actions(s)
        vals = np.ones([9]) * -2
        for a in actions:
            over = self.eval_helper(self.env.step(s.copy(), a, pid)[2], 3-pid, depth+1)[1]
            vals[a] = over
        return np.argmax(vals), -np.amax(vals)

    def evaluate_v(self, s, pid, active):
        if active:
            return -self.eval_helper(s, pid, 0)[1]
        else:
            return -self.evaluate_v(s, 3 - pid, True)
        
    def evaluate_p(self, s, pid):
        return self.eval_helper(s, pid, 0)[0]


class CardEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def evaluate_v(s):
        ss = copy.deepcopy(s)
        self_idx = ss['idx']
        done = False
        idx = ss['idx']
        r = 0
        while not done:
            idx = ss['idx']
            intention = np.array(to_char(env.Env.step_auto_static(card.Card.char2color(ss['player_cards'][ss['idx']]),
                                                                  np.array(to_value(ss['last_cards'] if ss['control_idx'] != idx else [])))))
            r, done = Pyenv.step_round(ss, intention)
        if idx == self_idx:
            return r
        else:
            return -r

        # curr_handcards = s['player_cards'][s['idx']]
        # return env.Env.get_cards_value(card.Card.char2color(curr_handcards))


class Node:
    def __init__(self, src, state, actions, priors):
        self.s = state
        self.a = actions
        self.src = src
        self.edges = []
        self.lock = threading.Lock()
        for i in range(self.a.size):
            self.edges.append(Edge(self, self.s, self.a[i], priors[i]))

    def choose(self, c):
        nsum_sqrt = math.sqrt(sum([e.n for e in self.edges]))
        cands = [e.q + c * e.p * nsum_sqrt / (1 + e.n) for e in self.edges]
        return self.edges[np.argmax(cands)]


class Edge:
    def __init__(self, src, state, action, prior):
        self.s = state
        self.a = action
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = prior
        self.src = src
        self.node = None


class MCTree:
    def __init__(self, s, agent, sess, *oppo_agents):
        self.env = Pyenv
        self.sess = sess
        self.agent = agent
        self.oppo_agents = oppo_agents
        subspace = Pyenv.get_actionspace(s)
        self.root = Node(None, s, subspace, agent.predict(s, subspace, sess))
        self.counter = 0
        self.counter_lock = threading.Lock()
        # self.evaluator = CardEvaluator

    def search(self, nthreads, n):
        self.counter = n
        threads = []
        for i in range(nthreads):
            t = threading.Thread(target=self.search_thread, args=())
            threads.append(t)
            t.start()
            sleep(0.05)
        for t in threads:
            t.join()

    def search_thread(self):
        while True:
            self.counter_lock.acquire()
            if self.counter == 0:
                self.counter_lock.release()
                break
            else:
                self.counter -= 1
            self.counter_lock.release()
            val, leaf = self.explore(self.root)
            if leaf:
                self.backup(leaf, val)
    
    def explore(self, node):
        node.lock.acquire()
        edge = node.choose(5.)
        sprime, r, done = self.env.step_static(edge.s, edge.a, self.sess, *self.oppo_agents)
        if done:
            if not edge.node:
                subspace = self.env.get_actionspace(sprime)
                edge.node = Node(edge, sprime, subspace, self.agent.predict(sprime, subspace, self.sess))
                node.lock.release()
                return r, edge.node
            else:
                # if we ran into this again, we'd like to reinforce our intuition
                node.lock.release()
                return r, edge.node
        if edge.node:
            node.lock.release()
            return self.explore(edge.node)
        else:
            subspace = self.env.get_actionspace(sprime)
            edge.node = Node(edge, sprime, subspace, self.agent.predict(sprime, subspace, self.sess))
            # we are in intermediate node, explore more
            if sprime.is_intermediate():
                node.lock.release()
                return self.explore(edge.node)
            node.lock.release()
            return self.agent.evaluate(sprime, self.sess), edge.node
        
    def backup(self, node, v):
        while node.src:
            node.lock.acquire()
            edge = node.src
            edge.n += 1
            edge.w += v
            edge.q = edge.w / edge.n
            node.lock.release()
            node = edge.src

    def play(self, temp):
        # print([e.n for e in self.root.edges])
        probs = np.array([pow(e.n, 1. / temp) for e in self.root.edges])
        probs = probs / np.sum(probs)
        # print(probs)
        return np.argmax(probs)
        # return np.random.choice(probs.size, p=probs)

    def give_cards_helper(self, node, temp, nactions):
        probs = np.zeros([nactions])
        valid_idx = [e.a for e in node.edges]
        for e in node.edges:
            probs[e.a] = pow(e.n, 1. / temp)
        probs = probs / np.sum(probs)

        # TODO: change max to sampling
        idx_max = np.random.choice(np.arange(len(valid_idx)), p=probs[valid_idx])
        # idx_max = np.argmax(probs[valid_idx])
        return node.edges[idx_max], probs

    # return mode, target distribution, intention
    def step(self, temp):
        distribution = {
            'decision_passive': np.zeros([4]),
            'bomb_passive': np.zeros([13]),
            'response_passive': np.zeros([14]),
            'decision_active': np.zeros([13]),
            'response_active': np.zeros([15]),
            'seq_length': np.zeros([12]),
            'minor_cards': [],
            'cards_history': []
        }

        s = self.root.s
        if s['stage'] == 'p_decision':
            decision_edge, distribution['decision_passive'] = self.give_cards_helper(self.root, temp, 4)
            decision = decision_edge.a
            if decision == 0:
                return 0, distribution, np.array([])
            elif decision == 1:
                bomb_edge, distribution['bomb_passive'] = self.give_cards_helper(decision_edge.node, temp, 13)
                return 1, distribution, np.array(to_char([bomb_edge.a + 3] * 4))
            elif decision == 2:
                return 0, distribution, np.array(['*', '$'])
            elif decision == 3:
                mode = 2
                response_edge, distribution['response_passive'] = self.give_cards_helper(decision_edge.node, temp, 14)
                node = response_edge.node
                bigger = response_edge.a + 1
                intention = give_cards_without_minor(bigger, np.array(
                    to_value(s['last_cards'] if s['control_idx'] != s['idx'] else [])), s['last_category_idx'], None)
                last_category_idx = s['last_category_idx']
                if last_category_idx == Category.THREE_ONE.value or \
                        last_category_idx == Category.THREE_TWO.value or \
                        last_category_idx == Category.THREE_ONE_LINE.value or \
                        last_category_idx == Category.THREE_TWO_LINE.value or \
                        last_category_idx == Category.FOUR_TWO.value:
                    mode += 5
                    minor_cards = []
                    distribution['cards_history'].append(to_char(intention))
                    while node.s['stage'] == 'minor':
                        minor_card_edge, minor_dist = self.give_cards_helper(node, temp, 15)
                        distribution['minor_cards'].append(minor_dist)
                        minor_card = minor_card_edge.a + 3
                        minor_cards.append(minor_card)
                        distribution['cards_history'].append([to_char(minor_card)])
                        if node.s['is_pair']:
                            minor_cards.append(minor_card)
                            distribution['cards_history'][-1].append(to_char(minor_card))
                        node = minor_card_edge.node
                    intention = np.concatenate([intention, minor_cards])
                return mode, distribution, np.array(to_char(intention))
        elif s['stage'] == 'a_decision':
            decision_edge, distribution['decision_active'] = self.give_cards_helper(self.root, temp, 13)
            active_category_idx = decision_edge.a + 1
            node = decision_edge.node

            # give response and go down once
            response_edge, distribution['response_active'] = self.give_cards_helper(node, temp, 15)
            node = response_edge.node

            mode = 3
            seq_length = None
            if active_category_idx == Category.SINGLE_LINE.value or \
                    active_category_idx == Category.DOUBLE_LINE.value or \
                    active_category_idx == Category.TRIPLE_LINE.value or \
                    active_category_idx == Category.THREE_ONE_LINE.value or \
                    active_category_idx == Category.THREE_TWO_LINE.value:
                seq_length_edge, distribution['seq_length'] = self.give_cards_helper(node, temp, 12)
                seq_length = seq_length_edge.a + 1
                node = seq_length_edge.node
                mode = 4
            intention = give_cards_without_minor(response_edge.a, np.array(
                to_value(s['last_cards'] if s['control_idx'] != s['idx'] else [])),
                                                 active_category_idx, seq_length)
            if active_category_idx == Category.THREE_ONE.value or \
                    active_category_idx == Category.THREE_TWO.value or \
                    active_category_idx == Category.THREE_ONE_LINE.value or \
                    active_category_idx == Category.THREE_TWO_LINE.value or \
                    active_category_idx == Category.FOUR_TWO.value:
                mode += 5
                minor_cards = []
                distribution['cards_history'].append(to_char(intention))
                while node.s['stage'] == 'minor':
                    minor_card_edge, minor_dist = self.give_cards_helper(node, temp, 15)
                    distribution['minor_cards'].append(minor_dist)
                    minor_card = minor_card_edge.a + 3
                    minor_cards.append(minor_card)
                    distribution['cards_history'].append([to_char(minor_card)])
                    if node.s['is_pair']:
                        minor_cards.append(minor_card)
                        distribution['cards_history'][-1].append(to_char(minor_card))
                    node = minor_card_edge.node
                intention = np.concatenate([intention, minor_cards])
            return mode, distribution, np.array(to_char(intention))

    def give_cards(self, temp):
        decision_idx, decision = self.give_cards_helper(self.root, temp)
        s = self.root.s
        if s['stage'] == 'p_decision':
            if decision == 0:
                return np.array([])
            elif decision == 1:
                _, bomb_response = self.give_cards_helper(self.root.edges[decision_idx].node, temp)
                return np.array(to_char([bomb_response + 3] * 4))
            elif decision == 2:
                return np.array(['*', '$'])
            elif decision == 3:
                _, bigger = self.give_cards_helper(self.root.edges[decision_idx].node, temp)
                bigger += 1
                intention = give_cards_without_minor(bigger, np.array(
                    to_value(s['last_cards'] if s['control_idx'] != s['idx'] else [])), s['last_category_idx'], None)
                return np.array(to_char(intention))
        elif s['stage'] == 'a_decision':
            active_category_idx = decision + 1
            node = self.root.edges[decision_idx].node
            # give response and go down once
            response_active_idx, response_active = self.give_cards_helper(node, temp)
            node = node.edges[response_active_idx].node

            seq_length = None
            if active_category_idx == Category.SINGLE_LINE.value or \
                    active_category_idx == Category.DOUBLE_LINE.value or \
                    active_category_idx == Category.TRIPLE_LINE.value or \
                    active_category_idx == Category.THREE_ONE_LINE.value or \
                    active_category_idx == Category.THREE_TWO_LINE.value:
                seq_length_idx, seq_length = self.give_cards_helper(node, temp)
                seq_length += 1
                node = node.edges[seq_length_idx].node
            intention = give_cards_without_minor(response_active, np.array(to_value(s['last_cards'] if s['control_idx'] != s['idx'] else [])),
                                                 active_category_idx, seq_length)
            if active_category_idx == Category.THREE_ONE.value or \
                    active_category_idx == Category.THREE_TWO.value or \
                    active_category_idx == Category.THREE_ONE_LINE.value or \
                    active_category_idx == Category.THREE_TWO_LINE.value or \
                    active_category_idx == Category.FOUR_TWO.value:
                minor_cards = []
                while node.s['stage'] == 'minor':
                    minor_card_idx, minor_card = self.give_cards_helper(node, temp)
                    minor_card += 3
                    minor_cards.append(minor_card)
                    if node.s['is_pair']:
                        minor_cards.append(minor_card)
                    node = node.edges[minor_card_idx].node
                intention = np.concatenate([intention, minor_cards])
            return np.array(to_char(intention))

import tensorflow as tf
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None])
    y = tf.squeeze(tf.Variable([[2.], [3.]]))
    loss = tf.reduce_sum(tf.square(x - y))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l = sess.run(loss, feed_dict={
            x: np.array([2, 2])
        })
        print(l)
    # pyenv = Pyenv()
    # for i in range(1):
    #     done = False
    #     pyenv.reset()
    #     pyenv.prepare()
    #     # pyenv.player_cards[0] = pyenv.player_cards[0][:-2]
    #     pyenv.player_cards[pyenv.lord_idx] = np.array(['$', '2', 'A', 'K', 'Q', 'J'])
    #     while not done:
    #         s = pyenv.dump_state()
    #         # s['player_cards'][s['idx']] = np.array(['5', '5', '5', '3'])
    #         mctree = MCTree(s)
    #         # print(mctree.root.a)
    #         try:
    #             mctree.search(1, 100)
    #         except ValueError:
    #             mctree.search(1, 1)
    #         # print(mctree.root.a)
    #         intention = mctree.give_cards(1.)
    #         print(pyenv.idx, pyenv.get_handcards())
    #         print('intention: ', intention)
    #         _, done = pyenv.step(intention)

    # env = Environment()
    # evltr = Evaluator(env)
    # s = np.zeros([3, 3])
    # s[0, 0] = 2
    # s[0, 1] = 1
    # s[0, 2] = 2
    # s[1, 0] = 2
    # s[1, 1] = 1
    # # s[1, 2] = 2
    # s[2, 0] = 1
    # print(env.stringify(s), end='\n\n')
    #
    # pid = 2
    # while env.is_over(s) == 0:
    #     if pid == 1:
    #         mctree = MCTree(env, evltr, s)
    #         mctree.search(1, 1000, 1)
    #         a = mctree.play(1.)
    #         _, _, s, _ = env.step(s, env.get_actions(s)[a], pid)
    #     else:
    #         a = int(input())
    #         _, _, s, _ = env.step(s, a, pid)
    #     pid = 3 - pid
    #     print(env.stringify(s), end='\n\n')



