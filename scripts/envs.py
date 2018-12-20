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
from tensorpack import *
from env import Env as CEnv
from TensorPack.MA_Hierarchical_Q.env import Env
from TensorPack.MA_Hierarchical_Q.predictor import Predictor
from TensorPack.MA_Hierarchical_Q.DQNModel import Model


weight_path = './model-500000'


class RandomEnv(Env):
    def step(self, intention):
        # print(self.get_curr_handcards())
        print(intention)
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
        agent_names = ['agent%d' % i for i in range(1, 4)]
        model = Model(agent_names, (100, 21, 256 + 256 * 2 + 120), 'Double', (100, 21), 0.99)
        self.predictors = {n: Predictor(OfflinePredictor(PredictConfig(
            model=model,
            session_init=SaverRestore(weight_path),
            input_names=[n + '/state', n + '_comb_mask', n + '/fine_mask'],
            output_names=[n + 'Qvalue']))) for n in self.get_all_agent_names()}

    def step(self, intention):
        print(intention)
        player, done = super().step(intention)
        if player != self.agent_names[0]:
            return 1, done
        else:
            return -1, done

    def step_auto(self):
        handcards = self.get_curr_handcards()
        last_two_cards = self.get_last_two_cards()
        prob_state = self.get_state_prob()
        intention = self.predictors[self.get_curr_agent_name()].predict(handcards, last_two_cards, prob_state)
        print(intention)
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
        print(intention)
        r, done, _ = self.step_manual(to_value(intention))
        return r, done

    def step_auto(self):
        intention, r, _ = super().step_auto()
        intention = to_char(intention)
        assert np.all(self.get_state_prob() >= 0) and np.all(self.get_state_prob() <= 1)
        print(intention)
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
