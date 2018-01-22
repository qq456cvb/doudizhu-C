import numpy as np
from card import Card, Category
from utils import to_char, to_value, get_mask_alter, give_cards_without_minor, \
    get_mask, action_space_single, action_space_pair, get_category_idx
import sys
import copy
import time
from statewrapper import WrappedState

sys.path.insert(0, './build/Release')
import env


def read_cards_input():
    intention = sys.stdin.readline()
    if intention == 'q':
        exit()
    intention = intention.split()
    return np.array(intention)


class Pyenv:
    total_cards = to_char(np.arange(3, 16)) * 4 + ['*', '$']

    def __init__(self):
        self.reset()

    def reset(self):
        self.histories = [[], [], []]
        self.player_cards = np.array([None, None, None])
        self.extra_cards = np.array([])
        self.lord_idx = 0
        self.land_score = 0
        self.control_idx = 0
        self.last_cards = np.array([])
        self.last_category_idx = 0
        self.idx = 0

    def get_role_ID(self):
        if self.idx == self.lord_idx:
            return 2
        if self.idx == (self.lord_idx + 2) % 3:
            return 1
        if self.idx == (self.lord_idx + 1) % 3:
            return 3

    def get_state(self):
        selfcards = Card.char2onehot(self.player_cards[self.idx])
        histories = [Card.char2onehot(self.histories[(self.idx + i) % 3]) for i in range(3)]
        total = np.ones([54])
        extra_cards = Card.char2onehot(self.extra_cards)
        remains = total - selfcards - histories[0] - histories[1] - histories[2]
        return np.concatenate([selfcards, remains, histories[0], histories[1], histories[2], extra_cards])

    @staticmethod
    def get_state_static(s):
        selfcards = Card.char2onehot(s['player_cards'][s['idx']])
        histories = [Card.char2onehot(s['histories'][(s['idx'] + i) % 3]) for i in range(3)]
        total = np.ones([54])
        extra_cards = Card.char2onehot(s['extra_cards'])
        remains = total - selfcards - histories[0] - histories[1] - histories[2]
        return np.concatenate([selfcards, remains, histories[0], histories[1], histories[2], extra_cards])

    def prepare(self):
        cards = np.array(Pyenv.total_cards.copy())
        np.random.shuffle(cards)
        for i in range(3):
            self.player_cards[i] = cards[i*17:(i+1)*17]
        self.extra_cards = cards[51:]
        vals = [env.Env.get_cards_value(Card.char2color(self.player_cards[i]))[0] for i in range(3)]
        self.lord_idx = np.argmax(vals)
        # distribute extra cards
        self.player_cards[self.lord_idx] = np.concatenate([self.player_cards[self.lord_idx], self.extra_cards])
        for i in range(3):
            self.player_cards[i] = np.array(sorted(list(self.player_cards[i]), key=lambda k: -Card.cards_to_value[k]))
        self.control_idx = self.lord_idx
        best_val = np.amax(vals)
        # call for score according to value
        if best_val > 20:
            self.land_score = 3
        elif best_val > 15:
            self.land_score = 2
        else:
            self.land_score = 1
        self.idx = self.lord_idx

    def get_handcards(self):
        return self.player_cards[self.idx]

    def get_last_outcards(self):
        if self.idx != self.control_idx:
            return self.last_cards
        else:
            return None

    def step(self, intention):
        idx = self.idx
        self.idx = (idx + 1) % 3
        if intention.size == 0:
            return 0, False

        self.last_category_idx = get_category_idx(intention)
        self.last_cards = intention
        self.control_idx = idx
        for card in intention:
            for i in range(self.player_cards[idx].size):
                if self.player_cards[idx][i] == card:
                    self.player_cards[idx] = np.delete(self.player_cards[idx], i)
                    break
        self.histories[idx] += list(intention)
        # test if it is bomb
        if intention.size == 4 and np.count_nonzero(intention == intention[0]) == 4:
            self.land_score *= 2

        if self.player_cards[idx].size == 0:
            self.idx = idx
            return self.land_score, True
        return 0, False

    def dump_state(self):
        s = WrappedState()
        for k, v in self.__dict__.items():
            s[k] = v
        s['stage'] = ''
        return s

    @staticmethod
    def step_round(s, intention):
        idx = s['idx']
        s['idx'] = (idx + 1) % 3
        if intention.size == 0:
            return 0, False

        s['last_category_idx'] = get_category_idx(intention)
        s['last_cards'] = intention
        s['control_idx'] = idx
        player_cards = s['player_cards']
        histories = s['histories']
        for card in intention:
            for i in range(player_cards[idx].size):
                if player_cards[idx][i] == card:
                    player_cards[idx] = np.delete(player_cards[idx], i)
                    break
        histories[idx] += list(intention)

        # test if it is bomb
        if intention.size == 4 and np.count_nonzero(intention == intention[0]) == 4:
            s['land_score'] *= 2

        if player_cards[idx].size == 0:
            s['idx'] = idx
            return s['land_score'], True
        return 0, False

    @staticmethod
    def step_helper(s, intention):
        idx_self = s['idx']
        r, done = Pyenv.step_round(s, intention)
        while not done and s['idx'] != idx_self:
            intention = np.array(to_char(env.Env.step_auto_static(Card.char2color(s['player_cards'][s['idx']]), np.array(to_value(s['last_cards'] if s['control_idx'] != s['idx'] else [])))))
            r, done = Pyenv.step_round(s, intention)
            # TODO: check if it's on the same team
            r = -r
        return s, r, done

    @staticmethod
    def step_static(s, a):
        sprime = copy.deepcopy(s)
        stage = s['stage']
        if stage == 'p_decision':
            if a == 0:
                sprime['stage'] = ''
                return Pyenv.step_helper(sprime, np.array([]))
            elif a == 1:
                sprime['stage'] = 'p_bomb'
                return sprime, 0, False
            elif a == 2:
                sprime['stage'] = ''
                return Pyenv.step_helper(sprime, np.array(['*', '$']))
            elif a == 3:
                sprime['stage'] = 'p_response'
                sprime['pending_stage'] = []
                last_category_idx = s['last_category_idx']
                if last_category_idx == Category.THREE_ONE.value or \
                        last_category_idx == Category.THREE_TWO.value or \
                        last_category_idx == Category.THREE_ONE_LINE.value or \
                        last_category_idx == Category.THREE_TWO_LINE.value or \
                        last_category_idx == Category.FOUR_TWO.value:
                    sprime['pending_stage'].append('minor')
                    sprime['dup_mask'] = np.zeros([15])
                    sprime['curr_minor_length'] = 0
                    if last_category_idx == Category.THREE_ONE.value or \
                            last_category_idx == Category.THREE_ONE_LINE.value or \
                            last_category_idx == Category.FOUR_TWO.value:
                        sprime['is_pair'] = False
                        sprime['dup_mask'] = get_mask(s['player_cards'][s['idx']], action_space_single, None)
                        if last_category_idx == Category.THREE_ONE.value:
                            sprime['minor_length'] = 1
                        elif last_category_idx == Category.FOUR_TWO.value:
                            sprime['minor_length'] = 2
                        else:
                            sprime['minor_length'] = len(s['last_cards']) // 4
                    else:
                        sprime['is_pair'] = True
                        sprime['dup_mask'][:13] = get_mask(s['player_cards'][s['idx']], action_space_pair, None)
                        if last_category_idx == Category.THREE_TWO.value:
                            sprime['minor_length'] = 1
                        else:
                            sprime['minor_length'] = len(s['last_cards']) // 5
                return sprime, 0, False
        elif stage == 'p_bomb':
            sprime['stage'] = ''
            return Pyenv.step_helper(sprime, np.array(to_char([a+3] * 4)))
        elif stage == 'p_response':
            sprime['stage'] = ''
            bigger = a + 1
            intention = give_cards_without_minor(bigger, np.array(to_value(s['last_cards'] if s['control_idx'] != s['idx'] else [])), s['last_category_idx'], None)
            pending_stage = sprime['pending_stage']
            if len(pending_stage) > 0:
                sprime['stage'] = pending_stage.pop(0)
                assert sprime['stage'] == 'minor'
                last_category_idx = s['last_category_idx']
                base = to_value(s['last_cards'])[0] - 3
                if last_category_idx == Category.THREE_ONE_LINE.value:
                    seq_length = len(s['last_cards']) // 4
                    for i in range(seq_length):
                        sprime['dup_mask'][base + bigger + i] = 0
                elif last_category_idx == Category.THREE_TWO_LINE.value:
                    seq_length = len(s['last_cards']) // 4
                    for i in range(seq_length):
                        sprime['dup_mask'][base + bigger + i] = 0
                else:
                    sprime['dup_mask'][base + bigger] = 0
                sprime['main_cards'] = to_char(intention)
                return sprime, 0, False
            return Pyenv.step_helper(sprime, np.array(to_char(intention)))
        elif stage == 'a_decision':
            active_category_idx = a + 1
            sprime['stage'] = 'a_response'
            sprime['pending_stage'] = []
            if active_category_idx == Category.SINGLE_LINE.value or \
                    active_category_idx == Category.DOUBLE_LINE.value or \
                    active_category_idx == Category.TRIPLE_LINE.value or \
                    active_category_idx == Category.THREE_ONE_LINE.value or \
                    active_category_idx == Category.THREE_TWO_LINE.value:
                sprime['pending_stage'].append('a_length')
            if active_category_idx == Category.THREE_ONE.value or \
                    active_category_idx == Category.THREE_TWO.value or \
                    active_category_idx == Category.THREE_ONE_LINE.value or \
                    active_category_idx == Category.THREE_TWO_LINE.value or \
                    active_category_idx == Category.FOUR_TWO.value:
                sprime['pending_stage'].append('minor')
                sprime['dup_mask'] = np.zeros([15])
                sprime['curr_minor_length'] = 0
                if active_category_idx == Category.THREE_ONE.value or \
                        active_category_idx == Category.THREE_ONE_LINE.value or \
                        active_category_idx == Category.FOUR_TWO.value:
                    sprime['is_pair'] = False
                    sprime['dup_mask'] = get_mask(s['player_cards'][s['idx']], action_space_single, None)
                    if active_category_idx == Category.THREE_ONE.value:
                        sprime['minor_length'] = 1
                    elif active_category_idx == Category.FOUR_TWO.value:
                        sprime['minor_length'] = 2
                else:
                    if active_category_idx == Category.THREE_TWO.value:
                        sprime['minor_length'] = 1
                    sprime['is_pair'] = True
                    sprime['dup_mask'][:13] = get_mask(s['player_cards'][s['idx']], action_space_pair, None)
            sprime['decision_active'] = a
            return sprime, 0, False
        elif stage == 'a_response':
            sprime['response_active'] = a
            pending_stage = sprime['pending_stage']
            if len(pending_stage) > 0:
                sprime['stage'] = pending_stage.pop(0)
                if sprime['stage'] == 'minor':
                    intention = give_cards_without_minor(a, np.array(to_value(s['last_cards'] if s['control_idx'] != s['idx'] else [])), s['decision_active'] + 1, 0)
                    sprime['main_cards'] = to_char(intention)
                    sprime['dup_mask'][a] = 0
                return sprime, 0, False
            else:
                sprime['stage'] = ''
                intention = give_cards_without_minor(a, np.array(to_value(s['last_cards'] if s['control_idx'] != s['idx'] else [])), s['decision_active'] + 1, 0)
                return Pyenv.step_helper(sprime, np.array(to_char(intention)))
        elif stage == 'a_length':
            pending_stage = sprime['pending_stage']
            if len(pending_stage) > 0:
                sprime['stage'] = pending_stage.pop(0)
                seq_length = a + 1
                response_active = sprime['response_active']
                intention = give_cards_without_minor(response_active, np.array(to_value(s['last_cards'] if s['control_idx'] != s['idx'] else [])),
                                                     s['decision_active'] + 1, seq_length)
                sprime['main_cards'] = to_char(intention)
                sprime['minor_length'] = seq_length
                for i in range(seq_length):
                    sprime['dup_mask'][response_active + i] = 0
                return sprime, 0, False
            else:
                sprime['stage'] = ''
                seq_length = a + 1
                response_active = sprime['response_active']
                intention = give_cards_without_minor(response_active, np.array(to_value(s['last_cards'] if s['control_idx'] != s['idx'] else [])),
                                                     s['decision_active'] + 1, seq_length)
                return Pyenv.step_helper(sprime, np.array(to_char(intention)))
        elif stage == 'minor':
            is_pair = s['is_pair']
            if s['curr_minor_length'] == 0:
                sprime['minor_cards'] = [to_char(a+3)] * (2 if is_pair else 1)
            else:
                sprime['minor_cards'] += [to_char(a+3)] * (2 if is_pair else 1)
            sprime['curr_minor_length'] += 1
            if sprime['curr_minor_length'] == sprime['minor_length']:
                sprime['stage'] = ''
                intention = np.concatenate([sprime['main_cards'], sprime['minor_cards']])
                return Pyenv.step_helper(sprime, np.array(intention))
            else:
                sprime['dup_mask'][a] = 0
                return sprime, 0, False

    @staticmethod
    def get_actionspace(s):
        stage = s['stage']
        control_idx = s['control_idx']
        idx = s['idx']
        last_cards_val = to_value(s['last_cards'] if s['control_idx'] != s['idx'] else [])
        player_cards = s['player_cards']
        curr_handcards = player_cards[idx]
        last_category_idx = s['last_category_idx']
        if stage == '':
            # clear dictionary
            s.clear()
            if control_idx == idx:
                s['stage'] = 'a_decision'
            else:
                s['stage'] = 'p_decision'
            stage = s['stage']
        if stage == 'p_decision':
            is_bomb = False
            if len(last_cards_val) == 4 and len(set(last_cards_val)) == 1:
                is_bomb = True
            decision_mask, response_mask, bomb_mask, _ = get_mask_alter(curr_handcards, s['last_cards'] if s['control_idx'] != s['idx'] else [], is_bomb, last_category_idx)
            s['response_mask'] = response_mask
            s['bomb_mask'] = bomb_mask
            return np.arange(4)[decision_mask == 1]
        elif stage == 'p_bomb':
            bomb_mask = s['bomb_mask']
            return np.arange(13)[bomb_mask == 1]
        elif stage == 'p_response':
            response_mask = s['response_mask']
            return np.arange(14)[response_mask == 1]
        elif stage == 'a_decision':
            decision_mask, response_mask, _, length_mask = get_mask_alter(curr_handcards, [], False, last_category_idx)
            s['response_mask'] = response_mask
            s['length_mask'] = length_mask
            return np.arange(13)[decision_mask == 1]
        elif stage == 'a_response':
            decision_active = s['decision_active']
            response_mask = s['response_mask']
            return np.arange(15)[response_mask[decision_active] == 1]
        elif stage == 'a_length':
            decision_active = s['decision_active']
            response_active = s['response_active']
            length_mask = s['length_mask']
            return np.arange(12)[length_mask[decision_active][response_active] == 1]
        elif stage == 'minor':
            dup_mask = s['dup_mask']
            return np.arange(15)[dup_mask == 1]

import tensorflow as tf, threading, time

class A:
    def test(self, nthreads):
        coord = tf.train.Coordinator()
        threads = []
        for i in range(nthreads):
            t = threading.Thread(target=self.search_thread, args=(i,))
            t.start()
            time.sleep(0.25)
            threads.append(t)
        coord.join(threads)

    def search_thread(self, i):
        print(i)


if __name__ == '__main__':
    a = A()
    a.test(4)
    # last_cards = np.array(['7', '7'])
    # curr_handcards = Card.char2color(np.array(['5', '6', '7', '7']))
    # print(env.Env.step_auto_static(curr_handcards, to_value(last_cards)))
    # pyenv = Pyenv()
    # pyenv.prepare()
    # done = False
    # idx = pyenv.lord_idx
    # while not done:
    #     print(pyenv.get_handcards(idx))
    #     intention = read_cards_input()
    #     _, done, idx = pyenv.step(intention, idx)

