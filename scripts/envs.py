from datetime import datetime
import numpy as np
from card import Card, Category, CardGroup, action_space
from utils import to_char, to_value, get_mask_alter, give_cards_without_minor, \
    get_mask, action_space_single, action_space_pair, get_category_idx, normalize
import sys
import os
if os.name == 'nt':
    sys.path.insert(0, '../build/Release')
else:
    sys.path.insert(0, '../build.linux')
from datetime import datetime
from env import Env as CEnv
from TensorPack.MA_Hierarchical_Q.env import Env


weight_path = ''


class RandomEnv(Env):
    def step(self, intention):
        # print(self.get_curr_handcards())
        # print(intention)
        player, done = super().step(intention)
        if player != self.agent_names[0]:
            return 1, done
        else:
            return -1, done

    def step_auto(self):
        mask = get_mask(self.get_curr_handcards(), action_space, self.get_last_outcards())
        intention = np.random.choice(action_space, 1, p=mask / mask.sum())[0]
        return self.step(intention)


class CDQNEnv(Env):
    def __init__(self, weight_path):
        super().__init__()

    def step(self, intention):
        player, done = super().step(intention)
        if player != self.agent_names[0]:
            return 1, done
        else:
            return -1, done

    def step_auto(self):
        mask = get_mask(self.get_curr_handcards(), action_space, self.get_last_outcards())
        intention = np.random.choice(action_space, 1, p=mask / mask.sum())[0]
        return self.step(intention)


class HCWBEnv(CEnv):
    def get_last_outcards(self):
        return to_char(super().get_last_outcards())

    def get_last_two_cards(self):
        last_two_cards = super().get_last_two_cards()
        last_two_cards = [to_char(c) for c in last_two_cards]
        return last_two_cards

    def get_curr_handcards(self):
        return to_char(super().get_curr_handcards())

    def step(self, intention):
        r, done, _ = self.step_manual(to_value(intention))
        return r, done

    def step_auto(self):
        _, r, _ = super().step_auto()
        return r, r != 0


def make_env(which):
    if which == 'HCWB':
        return HCWBEnv()
    elif which == 'RANDOM':
        return RandomEnv()
    elif which == 'CDQN':
        return CDQNEnv(weight_path)
    else:
        raise Exception('env type not supported')
