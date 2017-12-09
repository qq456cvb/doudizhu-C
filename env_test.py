import numpy as np
import card
import time
from collections import Counter
from card import Card, CardGroup, action_space, Category, action_space_category
import tensorflow as tf
import random
import struct
import sys
import a3c_alter_refined
sys.path.insert(0, './build/Release')

import env


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
    else:
        mask[0] = False
    return mask


# return <decision, response, minor cards> masks
def get_mask_alter(cards, last_cards, is_bomb, last_cards_category):
    decision_mask = None
    response_mask = None
    bomb_mask = None
    if not last_cards:
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
                diff = subspace[j][0] - last_cards[0]
                assert(diff > 0)
                response_mask[diff] = 1
                decision_mask[3] = 1
        if not is_bomb:
            bomb_mask = np.zeros([13])
            subspace = action_space_category[Category.QUADRIC.value]
            for j in range(len(subspace)):
                if counter_subset(subspace[j], cards):
                    bomb_mask[card.Card.char2value_3_17(subspace[j][0]) - 3] = 1
        return decision_mask, response_mask, bomb_mask


# return [3-17 value]
def give_cards_with_minor(response, minor_output, hand_cards_value, last_cards_value, category_idx):
    single_mask = np.zeros([15])
    for i in range(3, 18):
        if i in hand_cards_value:
            single_mask[i - 3] = 1

    double_mask = np.zeros([13])
    for i in range(3, 16):
        if counter_subset([i, i], hand_cards_value):
            double_mask[i - 3] = 1

    if last_cards_value:
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
            minor_output[np.where(single_mask == 0)] = 2
            return np.array([last_cards_value[0] + response] * 3 + [np.argmin(minor_output) + 3])
        elif category_idx == Category.THREE_TWO.value:
            double_mask[last_cards_value[0] + response - 3] = 0
            minor_output[np.where(double_mask == 0)] = 2
            return np.array([last_cards_value[0] + response] * 3 + [np.argmin(minor_output) + 3] * 2)
        elif category_idx == Category.SINGLE_LINE.value:
            return np.arange(last_cards_value[0] + response, last_cards_value[0] + response + len(last_cards_value))
        elif category_idx == Category.DOUBLE_LINE.value:
            link = np.arange(last_cards_value[0] + response, last_cards_value[0] + response + int(len(last_cards_value) / 2))
            return np.array([link, link]).T.reshape(-1)
        elif category_idx == Category.TRIPLE_LINE.value:
            link = np.arange(last_cards_value[0] + response, last_cards_value[0] + response + int(len(last_cards_value) / 3))
            return np.array([link, link, link]).T.reshape(-1)
        elif category_idx == Category.THREE_ONE_LINE.value:
            cnt = int(len(last_cards_value) / 4)
            for j in range(last_cards_value[0] + response, last_cards_value[0] + response + cnt):
                single_mask[j - 3] = 0
            link = np.arange(last_cards_value[0] + response, last_cards_value[0] + response + cnt)
            main = np.array([link, link, link]).T.reshape(-1)
            minor_output[np.where(single_mask == 0)] = 2
            minor = np.argsort(minor_output)[:cnt] + 3
            return np.concatenate([main, minor])
        elif category_idx == Category.THREE_TWO_LINE.value:
            cnt = int(len(last_cards_value) / 5)
            for j in range(last_cards_value[0] + response, last_cards_value[0] + response + cnt):
                double_mask[j - 3] = 0
            link = np.arange(last_cards_value[0] + response, last_cards_value[0] + response + cnt)
            main = np.array([link, link, link]).T.reshape(-1)
            minor_output[np.where(double_mask == 0)] = 2
            minor = np.argsort(minor_output)[:cnt] + 3
            minor = np.array([minor, minor]).T.reshape(-1)
            return np.concatenate([main, minor])
        elif category_idx == Category.FOUR_TWO.value:
            single_mask[last_cards_value[0] + response - 3] = 0
            minor_output[np.where(single_mask == 0)] = 2
            minor = np.argsort(minor_output)[:2] + 3
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
            minor_output[np.where(single_mask == 0)] = 2
            return np.array([response + 3] * 3 + [np.argmin(minor_output) + 3])
        elif category_idx == Category.THREE_TWO.value:
            double_mask[response] = 0
            minor_output[np.where(double_mask == 0)] = 2
            return np.array([response + 3] * 3 + [np.argmin(minor_output) + 3] * 2)
        elif category_idx == Category.SINGLE_LINE.value:
            # TODO : ambiguous here what the length should be, arbitrarily set to 5
            return np.arange(response + 3, response + 3 + 5)
        elif category_idx == Category.DOUBLE_LINE.value:
            link = np.arange(response + 3, response + 3 + 5)
            return np.array([link, link]).T.reshape(-1)
        elif category_idx == Category.TRIPLE_LINE.value:
            link = np.arange(response + 3, response + 3 + 5)
            return np.array([link, link, link]).T.reshape(-1)
        elif category_idx == Category.THREE_ONE_LINE.value:
            cnt = 5
            for j in range(response + 3, response + 3 + cnt):
                single_mask[j - 3] = 0
            link = np.arange(response + 3, response + 3 + cnt)
            main = np.array([link, link, link]).T.reshape(-1)
            minor_output[np.where(single_mask == 0)] = 2
            minor = np.argsort(minor_output)[:cnt] + 3
            return np.concatenate([main, minor])
        elif category_idx == Category.THREE_TWO_LINE.value:
            cnt = 5
            for j in range(response + 3, response + 3 + cnt):
                double_mask[j - 3] = 0
            link = np.arange(response + 3, response + 3 + cnt)
            main = np.array([link, link, link]).T.reshape(-1)
            minor_output[np.where(double_mask == 0)] = 2
            minor = np.argsort(minor_output)[:cnt] + 3
            minor = np.array([minor, minor]).T.reshape(-1)
            return np.concatenate([main, minor])
        elif category_idx == Category.FOUR_TWO.value:
            single_mask[response] = 0
            minor_output[np.where(single_mask == 0)] = 2
            minor = np.argsort(minor_output)[:2] + 3
            return np.array([response + 3] * 4 + [minor[0]] + [minor[1]])


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


def get_benchmark3(agents):
    e = env.Env()
    episodes = 0
    rewards = [0, 0, 0]
    total_episodes = 100
    i = 0
    while episodes < total_episodes:
        if episodes % 10 == 0:
            print('running %d' % episodes)
            print([rewards[i] / (episodes + 1) for i in range(3)])
        end = False
        e.reset()
        e.prepare()
        last_category_idx = 0
        idx = 0
        while not end:
            print("%d Current hand cards: " % idx, end='')
            print(a3c_alter_refined.to_char(e.get_curr_cards()))
            intention = agents[i].respond(e, last_category_idx)
            print("%d Intentions:" % idx, end='')
            print(a3c_alter_refined.to_char(intention))
            r, end, last_category_idx = e.step_manual(intention)
            rewards[i] += r
            i = (i + 1) % 3
        episodes += 1
    return [rewards[i] / total_episodes for i in range(3)]


class NaiveAgent:
    def __init__(self):
        pass

    def respond(self, env):
        curr_cards_value = env.get_curr_cards()
        curr_cards_char = a3c_alter_refined.to_char(curr_cards_value)
        last_cards_value = env.get_last_cards()
        last_cards_char = a3c_alter_refined.to_char(last_cards_value)
        
        mask = get_mask(curr_cards_char, action_space, last_cards_char)
        for i in range(len(action_space)):
            if mask[i]:
                # print('taking action, ', action_space[i])
                return env.step_manual(Card.char2value_3_17(action_space[i]))
        raise Exception("should not be here")


class RandomAgent:
    def __init__(self):
        pass

    def respond(self, env):
        curr_cards_value = env.get_curr_cards()
        curr_cards_char = a3c_alter_refined.to_char(curr_cards_value)
        last_cards_value = env.get_last_cards()
        last_cards_char = a3c_alter_refined.to_char(last_cards_value)

        mask = get_mask(curr_cards_char, action_space, last_cards_char)
        valid_actions = np.take(np.arange(len(action_space)), mask.nonzero())
        valid_actions = valid_actions.reshape(-1)
        a = np.random.choice(valid_actions)

        return Card.char2value_3_17(action_space[a])

class NetAgent:
    def __init__(self):
        self.network = a3c_alter_refined.CardNetwork(54 * 6, tf.train.AdamOptimizer(learning_rate=0.0001), "SLNetwork")
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, './Model/accuracy_bugfixed_lr0001_clipped/model.ckpt')

        self.action_space_single = action_space[1:16]
        self.action_space_pair = action_space[16:29]
        self.action_space_triple = action_space[29:42]
        self.action_space_quadric = action_space[42:55]

    def respond(self, env, last_category_idx):
        curr_cards_value = env.get_curr_cards()
        curr_cards_char = a3c_alter_refined.to_char(curr_cards_value)
        last_cards_value = env.get_last_cards()
        last_cards_char = a3c_alter_refined.to_char(last_cards_value)
        # mask = get_mask(curr_cards_char, action_space, last_cards_char)

        input_single = a3c_alter_refined.get_mask(curr_cards_char, self.action_space_single, None)
        input_pair = a3c_alter_refined.get_mask(curr_cards_char, self.action_space_pair, None)
        input_triple = a3c_alter_refined.get_mask(curr_cards_char, self.action_space_triple, None)
        input_quadric = a3c_alter_refined.get_mask(curr_cards_char, self.action_space_quadric, None)

        s = env.get_state()
        s = np.reshape(s, [1, -1])
        decision_passive_output, response_passive_output, bomb_passive_output, \
                    decision_active_output, response_active_output, minor_cards_output\
                     = self.sess.run([self.network.fc_decision_passive_output, 
                                self.network.fc_response_passive_output, self.network.fc_bomb_passive_output,
                                self.network.fc_decision_active_output, self.network.fc_response_active_output, 
                                self.network.fc_cards_value_output],
                        feed_dict = {
                            self.network.training: False,
                            self.network.input_state: s,
                            self.network.input_single: np.reshape(input_single, [1, -1]),
                            self.network.input_pair: np.reshape(input_pair, [1, -1]),
                            self.network.input_triple: np.reshape(input_triple, [1, -1]),
                            self.network.input_quadric: np.reshape(input_quadric, [1, -1])
                })
        if last_cards_value.size > 0:
            is_bomb = False
            if len(last_cards_value) == 4 and len(set(last_cards_value)) == 1:
                is_bomb = True
            decision_mask, response_mask, bomb_mask = get_mask_alter(curr_cards_char, last_cards_char, is_bomb, last_category_idx)
            decision_passive_output = decision_passive_output[0] * decision_mask
            decision_passive = np.argmax(decision_passive_output)
            if decision_passive == 0:
                return np.array([])
            elif decision_passive == 1:
                bomb_passive_output = bomb_passive_output[0] * bomb_mask
                return np.array([np.argmax(bomb_passive_output) + 3] * 4)
            elif decision_passive == 2:
                return np.array([16, 17])
            elif decision_passive == 3:
                response_passive_output = response_passive_output[0] * response_mask
                bigger = np.argmax(response_passive_output) + 1
                print(bigger)
                return np.array([])
        else:
            decision_mask, response_mask, _ = get_mask_alter(curr_cards_char, [], False, last_category_idx)
            decision_active_output = decision_active_output[0] * decision_mask
            decision_active = np.argmax(decision_active_output)
            response_active_output = decision_active_output[0] * response_mask[decision_active]
            print(response_active_output)
            return np.array([])
        return np.array([])
        

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
        val_self, _ = env.Env.get_cards_value(Card.char2color(self.agent_cards))
        self.oppo_cards = cards[int(len(cards) / 2):]
        val_oppo, _ = env.Env.get_cards_value(Card.char2color(self.oppo_cards))
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
def write_seq2(epochs, filename):
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
        enrivon = env.Env()
        random.shuffle(cards)
        for c in cards:
            if c == '10':
                c = '1'
            f.write(ord(c).to_bytes(1, byteorder='little', signed=False))
        # print(cards)
        handcards = [cards[:int(len(cards)/2)], cards[int(len(cards)/2):]]
        enrivon.reset()
        enrivon.prepare2_manual(Card.char2color(cards))
        end = False
        ind = 0
        while not end:
            intention, end = enrivon.step2_auto()
            put_list = Card.to_cards_from_3_17(intention)
            
            try:
                a = next(i for i, v in enumerate(action_space) if v == put_list)
            except StopIteration as e:
                print(put_list)
            
            
            f.write(a.to_bytes(2, byteorder='little', signed=False))
            # assert(action_space[a] == put_list)
            for c in put_list:
                handcards[ind].remove(c)
            ind = 1 - ind
        # assert((not handcards[0]) or (not handcards[1]))
        if i % 1000 == 0:
            print("writing %d..." % i)
            sys.stdout.flush()
    f.close()
    print("write completed with %d epochs" % epochs)


def read_seq2(filename):
    episodes = 0
    f = open(filename, 'rb')
    length = struct.unpack('H', f.read(2))[0]
    print(length)

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

def write_seq3(epochs, filename):
    f = open(filename, 'wb+')
    
    origin_cards = ['3', '3', '3', '3', '4', '4', '4', '4', '5', '5', '5', '5',
        '6', '6', '6', '6', '7', '7', '7', '7', '8', '8', '8', '8',
        '9', '9', '9', '9', '10', '10', '10', '10', 'J', 'J', 'J', 'J',
        'Q', 'Q', 'Q', 'Q', 'K', 'K', 'K', 'K', 'A', 'A', 'A', 'A',
        '2', '2', '2', '2', '*', '$']
    for i in range(epochs):
        cards = origin_cards.copy()
        enrivon = env.Env()
        lord_id = -1
        while lord_id == -1:
            random.shuffle(cards)
            enrivon.reset()
            lord_id = enrivon.prepare_manual(Card.char2color(cards))
        for c in cards:
            if c == '10':
                c = '1'
            f.write(ord(c).to_bytes(1, byteorder='little', signed=False))
        f.write(lord_id.to_bytes(2, byteorder='little', signed=False))
        handcards = [cards[:17], cards[17:34], cards[34:51]]
        extra_cards = cards[51:]
        handcards[lord_id] += extra_cards
        r = 0
        ind = lord_id
        while r == 0:
            intention, r = enrivon.step_auto()
            put_list = Card.to_cards_from_3_17(intention)
            # print(put_list)
            
            try:
                a = next(i for i, v in enumerate(action_space) if v == put_list)
            except StopIteration as e:
                print(put_list)
                # raise Exception('cards error')
            
            f.write(a.to_bytes(2, byteorder='little', signed=False))

            for c in put_list:
                handcards[ind].remove(c)
            ind = int(ind + 1) % 3
        f.write(r.to_bytes(2, byteorder='little', signed=True))
        
    f.close()
    print("write completed with %d epochs" % epochs)


def read_seq3(filename):
    episodes = 0
    f = open(filename, 'rb')

    eof = False
    while True:
        cards = []
        for i in range(54):
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
        lord_id = struct.unpack('H', f.read(2))[0]
        # print(cards)

        handcards = [cards[:17], cards[17:34], cards[34:51]]
        extra_cards = cards[51:]
        handcards[lord_id] += extra_cards
        r = 0
        ind = lord_id
        while r == 0:
            a = struct.unpack('H', f.read(2))[0]
            put_list = action_space[a]
            # print(put_list)
            for c in put_list:
                handcards[ind].remove(c)
            ind = int(ind + 1) % 3
            if (not handcards[0]) or (not handcards[1]) or (not handcards[2]):
                r = struct.unpack('h', f.read(2))[0]
        episodes += 1
        print(handcards)

if __name__ == "__main__":
    naive_agent = NaiveAgent()
    random_agent = RandomAgent()
    net_agent = NetAgent()
    print(get_benchmark3([net_agent, net_agent, net_agent]))
    # e = env.Env()
    # for i in range(1000):
    #     e.reset()
    #     e.prepare()
    #     r = 0
    #     last_cards = [0]
    #     j = 0
    #     while r == 0:
    #         j += 1
    #         intention, r, _ = e.step_auto()
    #
    #         if intention.size > 0 and len(last_cards) > 0:
    #             if intention[0] <= last_cards[0] and not (len(set(intention)) == 1 or len(set(last_cards)) == 1):
    #                 print(intention)
    #                 print(last_cards)
    #         last_cards = e.get_last_cards()
        # net_agent.respond(e, 0)
    # RandomAgent: card.Card.cards.copy() + card.Card.cards.copy() + card.Card.cards.copy() -0.6
    # e.prepare(Card.cards.copy() + Card.cards.copy())
    # print(e.agent_cards)
    # print(e.oppo_cards)
    # print(naive_agent.respond(env))
    # write_seq3(100, 'seq')
    # read_seq3('seq')

    # print(get_benchmark(['3', '3', '3', '3', '4', '4', '4', '4', '5', '5', '5', '5',
    #     '6', '6', '6', '6', '7', '7', '7', '7', '8', '8', '8', '8',
    #     '9', '9', '9', '9', '10', '10', '10', '10', 'J', 'J', 'J', 'J'], random_agent))
