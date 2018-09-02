import numpy as np
from card import Card, Category
from utils import to_char, to_value, get_mask_alter, give_cards_without_minor, \
    get_mask, action_space_single, action_space_pair, get_category_idx, normalize


class Env:
    total_cards = sorted(to_char(np.arange(3, 16)) * 4 + ['*', '$'], key=lambda k: Card.cards_to_value[k])

    def __init__(self, agent_names):
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
        self.curr_player = None

    def prepare(self):
        cards = Env.total_cards.copy()
        np.random.shuffle(cards)
        self.extra_cards = cards[17:20]
        self.player_cards[self.agent_names[0]] = cards[:20]
        self.player_cards[self.agent_names[1]] = cards[20:37]
        self.player_cards[self.agent_names[2]] = cards[37:]
        self.lord = self.agent_names[0]
        self.controller = self.lord
        self.curr_player = self.lord

    def step(self, intention):
        if len(intention) == 0:
            self.curr_player = self.agent_names[(self.agent_names.index(self.curr_player) + 1) % len(self.agent_names)]
            return self.curr_player, False
        else:
            self.last_cards_char = intention
            self.controller = self.curr_player
            for card in intention:
                self.player_cards[self.curr_player].remove(card)

            self.histories[self.curr_player].extend(intention)
            if len(self.player_cards) == 0:
                return self.curr_player, True
            else:
                self.curr_player = self.agent_names[
                    (self.agent_names.index(self.curr_player) + 1) % len(self.agent_names)]
                return self.curr_player, False

    def get_last_outcards(self):
        return self.last_cards_char if self.curr_player != self.controller else []

    def get_curr_handcards(self):
        return self.player_cards[self.curr_player]

    def get_state_prob(self):
        total_cards = np.ones([60])
        total_cards[53:56] = 0
        total_cards[57:60] = 0
        remain_cards = total_cards - Card.char2onehot60(self.get_curr_handcards()
                                                        + self.player_cards[self.agent_names[0]]
                                                        + self.player_cards[self.agent_names[1]]
                                                        + self.player_cards[self.agent_names[2]])
        next_cnt = len(self.player_cards[self.agent_names[(self.agent_names.index(self.curr_player) + 1) % len(self.agent_names)]])
        next_next_cnt = len(self.player_cards[self.agent_names[(self.agent_names.index(self.curr_player) + 2) % len(self.agent_names)]])
        right_prob_state = remain_cards * (next_cnt / (next_cnt + next_next_cnt))
        left_prob_state = remain_cards * (next_next_cnt / (next_cnt + next_next_cnt))
        prob_state = np.concatenate([right_prob_state, left_prob_state])
        return prob_state

