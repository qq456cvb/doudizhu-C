import numpy as np
import card
import time
from collections import Counter
from card import Card, CardGroup, action_space
import random
import struct
import sys
sys.path.insert(0, './build/Release')

import env

def counter_subset(list1, list2):
    c1, c2 = Counter(list1), Counter(list2)

    for (k, n) in c1.items():
        if n > c2[k]:
            return False
    return True


def get_benchmark(cards, target):
    env = Env()
    episodes = 0
    rewards = 0
    total_episodes = 500
    while episodes < total_episodes:
        if episodes % 100 == 0:
            print('running %d' % episodes)
            print(rewards / (episodes + 1))
        end = False
        env.reset()
        env.prepare(cards)
        while not end:
            r, end = target.respond(env)
            rewards += r
        if r == 1.:
            print('you win!')
        else:
            print('you lose!')
        episodes += 1
    return rewards / total_episodes

class NaiveAgent:
    def __init__(self):
        pass

    def respond(self, env):
        mask = env.get_mask()
        for i in range(len(action_space)):
            if mask[i]:
                # print('taking action, ', action_space[i])
                return env.step(action_space[i])
        raise Exception("should not be here") 
        return None, None

class RandomAgent:
    def __init__(self):
        pass

    def respond(self, env):
        mask = env.get_mask()
        valid_actions = np.take(np.arange(len(action_space)), mask.nonzero())
        valid_actions = valid_actions.reshape(-1)
        a = np.random.choice(valid_actions)

        # print('taking action, ', action_space[a])
        return env.step(action_space[a])

class Env:
    def __init__(self):
        self.agent_cards = []
        self.oppo_cards = []
        self.controller = 0
        self.last_cards = []
        self.history = [[], []]
        self.cards_cache = []

    def get_state(self):
        return np.vstack([Card.to_onehot(self.agent_cards),
            Card.to_onehot(self.oppo_cards),
            Card.to_onehot(self.history[0]),
            Card.to_onehot(self.history[1])])

    def step(self, intention):
        if not intention:
            self.controller = 1
            for a in action_space:
                if not a:
                    continue
                if counter_subset(a, self.oppo_cards):
                    self.last_cards = a
                    
                    group = CardGroup.to_cardgroup(a)
                    for card in a:
                        self.oppo_cards.remove(card)
                        self.history[1].append(card)
                    if not self.oppo_cards:
                        return -1, True
                    return 0, False

        self.controller = 0
        self.last_cards = intention
        for card in intention:
            self.agent_cards.remove(card)
            self.history[0].append(card)
        if not self.agent_cards:
            return 1, True

        group_intention = CardGroup.to_cardgroup(intention)
        for a in action_space:
            if not a:
                continue
            if counter_subset(a, self.oppo_cards):
                group = CardGroup.to_cardgroup(a)
                if group.bigger_than(group_intention):
                    for card in a:
                        self.oppo_cards.remove(card)
                        self.history[1].append(card)
                    self.last_cards = a
                    self.controller = 1
                    break
        if not self.oppo_cards:
            return -1, True
        

        return 0, False

    def reset(self):
        self.agent_cards = []
        self.oppo_cards = []
        self.history = [[], []]
        self.last_cards = []
        self.controller = 0
   
    def prepare(self, cards):
        if not self.cards_cache:
            random.seed(44)
            random.shuffle(cards)
        else:
            cards = self.cards_cache
        self.agent_cards = cards[:int(len(cards) / 2)]
        val_self, _ = env.Env.get_cards_value(Card.onehot2color(Card.to_onehot(self.agent_cards)))
        self.oppo_cards = cards[int(len(cards) / 2):]
        val_oppo, _ = env.Env.get_cards_value(Card.onehot2color(Card.to_onehot(self.oppo_cards)))
        print(val_self, val_oppo)
        if (val_self < val_oppo + 5):
            self.prepare(cards)
        else:
            self.cards_cache = cards
    
    def get_mask(self):
        # 1 valid; 0 invalid
        mask = np.zeros_like(action_space)
        for j in range(mask.size):
            if counter_subset(action_space[j], self.agent_cards):
                mask[j] = 1
        mask = mask.astype(bool)
        if self.controller == 1:
            mask[0] = True
            for j in range(1, mask.size):
                if mask[j] and not card.CardGroup.to_cardgroup(action_space[j]).\
                        bigger_than(card.CardGroup.to_cardgroup(self.last_cards)):
                    mask[j] = False
        else:
            mask[0] = False
        return mask


# number of cards, [[cards as char], [action index]]
def write_seq(epochs, filename):
    f = open(filename, 'wb+')
    # origin_cards = ['3', '3', '3', '3', '4', '4', '4', '4', '5', '5', '5', '5',
    #     '6', '6', '6', '6', '7', '7', '7', '7', '8', '8', '8', '8',
    #     '9', '9', '9', '9', '10', '10', '10', '10', 'J', 'J', 'J', 'J',
    #     'Q', 'Q', 'Q', 'Q', 'K', 'K', 'K', 'K', 'A', 'A', 'A', 'A',
    #     '2', '2', '2', '2', '*', '$']
    origin_cards = ['3', '3', '3', '3', '4', '4', '4', '4', '5', '5', '5', '5',
        '6', '6', '6', '6', '7', '7', '7', '7', '8', '8', '8', '8',
        '9', '9', '9', '9', '10', '10', '10', '10', 'J', 'J', 'J', 'J']
    f.write(len(origin_cards).to_bytes(2, byteorder='little', signed=False))
    for i in range(epochs):
        cards = origin_cards.copy()
        random.shuffle(cards)
        for c in cards:
            if c == '10':
                c = '1'
            f.write(ord(c).to_bytes(1, byteorder='little', signed=False))
        # print(cards)
        handcards = [cards[:int(len(cards)/2)], cards[int(len(cards)/2):]]
        e.reset()
        e.prepare2_manual(Card.char2color(cards))
        end = False
        ind = 0
        while not end:
            intention, end = e.step2_auto()
            put_list = Card.to_cards_from_3_17(intention)
            # print(put_list)
            a = next(i for i, v in enumerate(action_space) if v == put_list)
            
            f.write(a.to_bytes(2, byteorder='little', signed=False))
            # assert(action_space[a] == put_list)
            for c in put_list:
                handcards[ind].remove(c)
            ind = 1 - ind
        # assert((not handcards[0]) or (not handcards[1]))
    f.close()
    print("write completed with %d epochs" % epochs)


def read_seq(filename):
    episodes = 0
    f = open(filename, 'rb')
    length = struct.unpack('H', f.read(2))[0]
    print(length)
    e = env.Env()

    eof = False
    while True:
        cards = []
        for i in range(length):
            b = f.read(1)
            if not b:
                eof = True
                break
            c = str((struct.unpack('c', b)[0]).decode('ascii'))
            
            if c == '1':
                c = '10'
            cards.append(c)
        
        if eof:
            break
        # print(cards)

        handcards = [cards[:int(len(cards)/2)], cards[int(len(cards)/2):]]
        end = False
        ind = 0
        while not end:
            a = struct.unpack('H', f.read(2))[0]
            put_list = action_space[a]
            # print(put_list)
            for c in put_list:
                handcards[ind].remove(c)
            ind = 1 - ind
            if (not handcards[0]) or (not handcards[1]):
                end = True
        episodes += 1


if __name__ == "__main__":
    naive_agent = NaiveAgent()
    random_agent = RandomAgent()
    # RandomAgent: card.Card.cards.copy() + card.Card.cards.copy() + card.Card.cards.copy() -0.6
    e = env.Env()
    # e.prepare(Card.cards.copy() + Card.cards.copy())
    # print(e.agent_cards)
    # print(e.oppo_cards)
    # print(naive_agent.respond(env))
    write_seq(5, 'seq')
    read_seq('seq')

    # print(get_benchmark(['3', '3', '3', '3', '4', '4', '4', '4', '5', '5', '5', '5',
    #     '6', '6', '6', '6', '7', '7', '7', '7', '8', '8', '8', '8',
    #     '9', '9', '9', '9', '10', '10', '10', '10', 'J', 'J', 'J', 'J'], random_agent))
