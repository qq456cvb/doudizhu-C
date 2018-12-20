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


class Agent:
    def __init__(self, role_id):
        self.role_id = role_id

    def intention(self, env):
        pass


class RandomAgent(Agent):
    def intention(self, env):
        mask = get_mask(env.get_curr_handcards(), action_space, env.get_last_outcards())
        intention = np.random.choice(action_space, 1, p=mask / mask.sum())[0]
        return intention


class HCWBAgent(Agent):
    def intention(self, env):
        intention = CEnv.step_auto_static(Card.char2onehot60(env.get_curr_handcards()), to_value(env.get_last_outcards()))
        return intention


class CDQNAgent(Agent):
    def __init__(self, role_id, weight_path):
        super().__init__(role_id)

    def intention(self, env):
        pass


def make_agent(which, role_id):
    if which == 'HCWB':
        return HCWBAgent(role_id)
    elif which == 'RANDOM':
        return RandomAgent(role_id)
    elif which == 'CDQN':
        return CDQNAgent(role_id, weight_path)
    else:
        raise Exception('env type not supported')

