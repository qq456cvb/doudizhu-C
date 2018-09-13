from datetime import datetime
import numpy as np
from card import Card, Category
from TensorPack.MA_Hierarchical_Q.predictor import Predictor
from utils import to_char, to_value, get_mask_alter, give_cards_without_minor, \
    get_mask, action_space_single, action_space_pair, get_category_idx, normalize


class Env:
    total_cards = sorted(to_char(np.arange(3, 16)) * 4 + ['*', '$'], key=lambda k: Card.cards_to_value[k])

    def __init__(self, agent_names):
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
        self.curr_player = None

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

    def get_curr_handcards(self):
        return self.player_cards[self.curr_player].copy()

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


if __name__ == '__main__':
    env = Env(['1', '2', '3'])
    predictors = {n: Predictor(lambda x: [np.random.rand(1, 21)]) for n in env.get_all_agent_names()}
    for _ in range(1000):
        env.reset()
        env.prepare()
        done = False
        while not done:
            handcards = env.get_curr_handcards()
            last_cards = env.get_last_outcards()
            prob_state = env.get_state_prob()
            action = predictors[env.get_curr_agent_name()].predict(handcards, last_cards, prob_state)
            _, done = env.step(action)


