from datetime import datetime
import numpy as np
from card import Card, Category, CardGroup, action_space
from utils import to_char, to_value, get_mask_alter, give_cards_without_minor, \
    get_mask, action_space_single, action_space_pair, get_category_idx, normalize
import sys
import os
if os.name == 'nt':
    sys.path.insert(0, './build/Release')
else:
    sys.path.insert(0, './build.linux')
from datetime import datetime
from tensorpack import *
from env import Env as CEnv
from mct import mcsearch, CCard, CCardGroup, CCategory
from TensorPack.MA_Hierarchical_Q.predictor import Predictor
from TensorPack.MA_Hierarchical_Q.DQNModel import Model


weight_path = './model-302500'


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


class RHCPAgent(Agent):
    def intention(self, env):
        intention = to_char(CEnv.step_auto_static(Card.char2color(env.get_curr_handcards()), to_value(env.get_last_outcards())))
        return intention


class CDQNAgent(Agent):
    def __init__(self, role_id, weight_path):
        def role2agent(role):
            if role == 2:
                return 'agent1'
            elif role == 1:
                return 'agent3'
            else:
                return 'agent2'
        super().__init__(role_id)
        agent_names = ['agent%d' % i for i in range(1, 4)]
        model = Model(agent_names, (1000, 21, 256 + 256 * 2 + 120), 'Double', (1000, 21), 0.99)
        self.predictor = Predictor(OfflinePredictor(PredictConfig(
            model=model,
            session_init=SaverRestore(weight_path),
            input_names=[role2agent(role_id) + '/state', role2agent(role_id) + '_comb_mask', role2agent(role_id) + '/fine_mask'],
            output_names=[role2agent(role_id) + '/Qvalue'])), num_actions=(1000, 21))

    def intention(self, env):
        handcards = env.get_curr_handcards()
        last_two_cards = env.get_last_two_cards()
        prob_state = env.get_state_prob()
        intention = self.predictor.predict(handcards, last_two_cards, prob_state)
        return intention


class MCTAgent(Agent):
    def intention(self, env):
        def char2ccardgroup(chars):
            cg = CardGroup.to_cardgroup(chars)
            ccg = CCardGroup([CCard(to_value(c) - 3) for c in cg.cards], CCategory(cg.type), cg.value, cg.len)
            return ccg

        def ccardgroup2char(cg):
            return [to_char(int(c) + 3) for c in cg.cards]
        handcards_char = env.get_curr_handcards()
        chandcards = [CCard(to_value(c) - 3) for c in handcards_char]
        player_idx = env.get_current_idx()
        unseen_cards = env.player_cards[env.agent_names[(player_idx + 1) % 3]] + env.player_cards[env.agent_names[(player_idx + 2) % 3]]
        cunseen_cards = [CCard(to_value(c) - 3) for c in unseen_cards]

        # print(env.player_cards)
        next_handcards_cnt = len(env.player_cards[env.agent_names[(player_idx + 1) % 3]])

        last_cg = char2ccardgroup(env.get_last_outcards())

        # print(handcards_char, env.get_last_outcards(), next_handcards_cnt, env.curr_player, env.controller, env.lord)
        caction = mcsearch(chandcards, cunseen_cards, next_handcards_cnt, last_cg,
                           (env.agent_names.index(env.curr_player) - env.agent_names.index(env.lord) + 3) % 3,
                           (env.agent_names.index(env.controller) - env.agent_names.index(env.lord) + 3) % 3, 10, 50, 500)
        intention = ccardgroup2char(caction)
        return intention


def make_agent(which, role_id):
    if which == 'RHCP':
        return RHCPAgent(role_id)
    elif which == 'RANDOM':
        return RandomAgent(role_id)
    elif which == 'CDQN':
        return CDQNAgent(role_id, weight_path)
    elif which == 'MCT':
        return MCTAgent(role_id)
    else:
        raise Exception('env type not supported')

