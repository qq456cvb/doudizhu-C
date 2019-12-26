import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))
from datetime import datetime
import numpy as np
from card import Card, Category, CardGroup
from utils import to_char, to_value, get_mask_alter, give_cards_without_minor, \
    get_mask, action_space_single, action_space_pair, get_category_idx, normalize

from datetime import datetime
from env import Env as CEnv


class Env:
    total_cards = sorted(to_char(np.arange(3, 16)) * 4 + ['*', '$'], key=lambda k: Card.cards_to_value[k])

    def __init__(self, agent_names=('agent1', 'agent2', 'agent3')):
        seed = (id(self) + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
        np.random.seed(seed)
        self.agent_names = agent_names
        self.reset()

    def get_all_agent_names(self):
        return self.agent_names

    def get_curr_agent_name(self):
        return self.curr_player

    def reset(self):
        self.histories = {n: [] for n in self.agent_names}
        self.player_cards = {n: [] for n in self.agent_names}
        self.extra_cards = []
        self.lord = None
        self.controller = None
        self.last_cards_char = []
        self.out_cards = [[] for _ in range(3)]
        self.curr_player = None

    def get_role_ID(self):
        curr_idx = self.get_current_idx()
        assert 0 <= curr_idx <= 2
        if curr_idx == 0:
            return 2
        if curr_idx == 1:
            return 3
        return 1

    def get_current_idx(self):
        return self.agent_names.index(self.curr_player)

    def prepare(self):
        cards = Env.total_cards.copy()
        np.random.shuffle(cards)
        self.extra_cards = cards[17:20]
        self.player_cards[self.agent_names[0]] = sorted(cards[:20], key=lambda k: Card.cards_to_value[k])
        self.player_cards[self.agent_names[1]] = sorted(cards[20:37], key=lambda k: Card.cards_to_value[k])
        self.player_cards[self.agent_names[2]] = sorted(cards[37:], key=lambda k: Card.cards_to_value[k])
        self.lord = self.agent_names[0]
        self.controller = self.lord
        self.curr_player = self.lord

    def step(self, intention):
        self.out_cards[self.agent_names.index(self.curr_player)] = intention
        if len(intention) == 0:
            self.curr_player = self.agent_names[(self.agent_names.index(self.curr_player) + 1) % len(self.agent_names)]
            return self.curr_player, False
        else:
            self.last_cards_char = intention
            self.controller = self.curr_player
            for card in intention:
                self.player_cards[self.curr_player].remove(card)

            self.histories[self.curr_player].extend(intention)
            if len(self.player_cards[self.curr_player]) == 0:
                return self.curr_player, True
            else:
                self.curr_player = self.agent_names[
                    (self.agent_names.index(self.curr_player) + 1) % len(self.agent_names)]
                return self.curr_player, False

    def get_last_outcards(self):
        return self.last_cards_char.copy() if self.curr_player != self.controller else []

    def get_last_two_cards(self):
        return [self.out_cards[(self.agent_names.index(self.curr_player) + 2) % len(self.agent_names)].copy(),
                self.out_cards[(self.agent_names.index(self.curr_player) + 1) % len(self.agent_names)].copy()]

    def get_curr_handcards(self):
        return self.player_cards[self.curr_player].copy()

    def get_state_prob(self):
        total_cards = np.ones([60])
        total_cards[53:56] = 0
        total_cards[57:60] = 0
        player_idx = self.get_current_idx()
        remain_cards = total_cards - Card.char2onehot60(self.get_curr_handcards()
                                                        + self.histories[self.agent_names[player_idx]]
                                                        + self.histories[self.agent_names[(player_idx + 1) % 3]]
                                                        + self.histories[self.agent_names[(player_idx + 2) % 3]])
        # sanity check
        # remain_cards_check = Card.char2onehot60(self.player_cards[self.agent_names[(player_idx + 1) % 3]] + self.player_cards[self.agent_names[(player_idx + 2) % 3]])
        # remain_cards_cp = remain_cards.copy()
        # normalize(remain_cards_cp, 0, 60)
        # assert np.all(remain_cards_cp == remain_cards_check)
        next_cnt = len(self.player_cards[self.agent_names[(player_idx + 1) % len(self.agent_names)]])
        next_next_cnt = len(self.player_cards[self.agent_names[(player_idx + 2) % len(self.agent_names)]])
        right_prob_state = remain_cards * (next_cnt / (next_cnt + next_next_cnt))
        left_prob_state = remain_cards * (next_next_cnt / (next_cnt + next_next_cnt))
        prob_state = np.concatenate([right_prob_state, left_prob_state])
        return prob_state


def char2ccardgroup(chars):
    cg = CardGroup.to_cardgroup(chars)
    ccg = CCardGroup([CCard(to_value(c) - 3) for c in cg.cards], CCategory(cg.type), cg.value, cg.len)
    return ccg


def ccardgroup2char(cg):
    return [to_char(int(c) + 3) for c in cg.cards]


import time
if __name__ == '__main__':
    # cg = CardGroup.to_cardgroup(['3', '3', '4', '4', '5', '5'])
    # print(cg.len)
    # print(cg.type)
    # ccg = CCardGroup([CCard(to_value(c) - 3) for c in cg.cards], CCategory(cg.type), cg.value, cg.len)

    cnt = {
        '1': 0,
        '2': 0,
        '3': 0
    }

    # python env usage
    env = Env(['1', '2', '3'])
    agent_names = ['1', '2', '3']

    for _ in range(1):
        env.reset()
        env.prepare()
        done = False
        while not done:
            print('here')
            handcards = env.get_curr_handcards()

            env.get_state_prob()
            t = time.perf_counter()
            chandcards = [CCard(to_value(c) - 3) for c in handcards]
            unseen_cards = env.player_cards[agent_names[(env.get_current_idx() + 1) % len(env.agent_names)]].copy() \
                            + env.player_cards[agent_names[(env.get_current_idx() + 2) % len(env.agent_names)]].copy()
            print('here')
            cunseen_cards = [CCard(to_value(c) - 3) for c in unseen_cards]
            print('here')
            next_handcards_cnt = len(env.player_cards[agent_names[(env.get_current_idx() + 1) % len(env.agent_names)]])

            last_cg = char2ccardgroup(env.get_last_outcards())
            caction = mcsearch(chandcards, cunseen_cards, next_handcards_cnt, last_cg, env.agent_names.index(env.curr_player), env.agent_names.index(env.controller))

            action = ccardgroup2char(caction)
            print(action)
            winner, done = env.step(action)
            if done:
                for agent_name in agent_names:
                    if agent_name == winner:
                        cnt[agent_name] += 1
                        print(agent_name, ' wins')
                    else:
                        if env.get_all_agent_names().index(winner) + env.get_all_agent_names().index(agent_name) == 3:
                            cnt[agent_name] += 1
                            print(agent_name, winner, ' all wins')
    print(cnt)

    # C env usage
    # TODO
    env = CEnv()
    # for _ in range(1):
    #     env.reset()
    #     env.prepare()
    #     done = False
    #     while not done:
    #         handcards = to_char(env.get_curr_handcards())
    #
    #         chandcards = [CCard(to_value(c) - 3) for c in handcards]
    #         unseen_cards = env.player_cards[agent_names[(env.get_current_idx() + 1) % len(env.agent_names)]].copy() \
    #                         + env.player_cards[agent_names[(env.get_current_idx() + 2) % len(env.agent_names)]].copy()
    #         cunseen_cards = [CCard(to_value(c) - 3) for c in unseen_cards]
    #         next_handcards_cnt = len(env.player_cards[agent_names[(env.get_current_idx() + 1) % len(env.agent_names)]])
    #
    #         last_cg = char2ccardgroup(env.get_last_outcards())
    #         caction = mcsearch(chandcards, cunseen_cards, next_handcards_cnt, last_cg, env.agent_names.index(env.curr_player), env.agent_names.index(env.controller))
    #
    #         action = ccardgroup2char(caction)
    #         print(action)
    #         winner, done = env.step(action)
    #         if done:
    #             for agent_name in agent_names:
    #                 if agent_name == winner:
    #                     cnt[agent_name] += 1
    #                     print(agent_name, ' wins')
    #                 else:
    #                     if env.get_all_agent_names().index(winner) + env.get_all_agent_names().index(agent_name) == 3:
    #                         cnt[agent_name] += 1
    #                         print(agent_name, winner, ' all wins')
    # print(cnt)


