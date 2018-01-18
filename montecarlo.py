import numpy as np
import math
from pyenv import Pyenv
import card
from utils import to_char, to_value
import copy

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
    def __init__(self, src, state, actions):
        self.s = state
        self.a = actions
        self.src = src
        self.edges = [Edge(self, self.s, a, 0) for a in self.a]

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
    def __init__(self, s):
        self.env = Pyenv
        self.root = Node(None, s, Pyenv.get_actionspace(s))
        self.evaluator = CardEvaluator

    def search(self, nthreads, n):
        for i in range(n):
            val, leaf = self.explore(self.root)
            if leaf:
                self.backup(leaf, val)
    
    def explore(self, node):
        edge = node.choose(0)
        sprime, r, done = self.env.step_static(edge.s, edge.a)
        if done:
            if not edge.node:
                edge.node = Node(edge, sprime, self.env.get_actionspace(sprime))
                return r, edge.node
            else:
                # if we ran into this again, we want to reinforce our intuition
                return r, edge.node
        if edge.node:
            return self.explore(edge.node)
        else:
            edge.node = Node(edge, sprime, self.env.get_actionspace(sprime))
            # we are in intermediate node, explore more
            if sprime.is_intermediate():
                return self.explore(edge.node)
            return self.evaluator.evaluate_v(sprime), edge.node
        
    def backup(self, node, v):
        while node.src:
            edge = node.src
            edge.n += 1
            edge.w += v
            edge.q = edge.w / edge.n
            node = edge.src

    def play(self, temp):
        # print([e.n for e in self.root.edges])
        probs = np.array([pow(e.n, 1. / temp) for e in self.root.edges])
        probs = probs / np.sum(probs)
        print(probs)
        return np.argmax(probs)
        # return np.random.choice(probs.size, p=probs)


if __name__ == '__main__':
    pyenv = Pyenv()
    pyenv.prepare()
    s = pyenv.dump_state()
    s['player_cards'][s['idx']] = np.array(['5', '5'])
    mctree = MCTree(s)
    mctree.search(1, 1000)
    mctree.play(1.)

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



