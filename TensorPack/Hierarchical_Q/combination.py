import numpy as np
from pyenv import Pyenv
import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))
from env import get_combinations_recursive, get_combinations_nosplit
from utils import get_mask_onehot60, get_mask
from card import action_space, clamp_action_idx, Card, action_space_category, Category, augment_action_space_onehot60, CardGroup, augment_action_space
from tensorpack.utils.stats import StatCounter


def dancing_link():
    env = Pyenv()
    env.reset()
    env.prepare()
    # print(env.get_handcards())
    cards = env.get_handcards()
    cards = ['3', '3', '3', '4', '4', '4']
    import timeit
    begin = timeit.default_timer()
    card_mask = Card.char2onehot60(cards).astype(np.uint8)
    # mask = get_mask_onehot60(cards, action_space, None).astype(np.uint8)
    last_cards = ['3', '3']
    mask = augment_action_space_onehot60
    a = np.expand_dims(1 - card_mask, 0) * mask
    row_idx = set(np.where(a > 0)[0])

    # tmp = np.ones(len(augment_action_space))
    # tmp[row_idx] = 0
    # tmp[0] = 0
    # valid_row_idx = np.where(tmp > 0)[0]
    valid_row_idx = [i for i in range(1, len(augment_action_space)) if i not in row_idx]
    idx_must_be_contained = set([idx for idx in valid_row_idx if CardGroup.to_cardgroup(augment_action_space[idx]).\
                    bigger_than(CardGroup.to_cardgroup(last_cards))])
    print(idx_must_be_contained)
    mask = mask[valid_row_idx, :]
    idx_mapping = dict(zip(range(mask.shape[0]), valid_row_idx))

    # augment mask
    # TODO: known issue: 555444666 will not decompose into 5554 and 66644

    combs = get_combinations_nosplit(mask, Card.char2onehot60(cards).astype(np.uint8))
    combs = [[clamp_action_idx(idx_mapping[idx]) for idx in comb] for comb in combs]
    combs = [comb for comb in combs if not idx_must_be_contained.isdisjoint(comb)]
    fine_mask = np.zeros([len(combs), 21])
    for i in range(len(combs)):
        for j in range(len(combs[i])):
            if combs[i][j] in idx_must_be_contained:
                fine_mask[i][j] = 1
    print(fine_mask)
    end = timeit.default_timer()
    print(end - begin)

    print(len(combs))
    for comb in combs:
        for idx in comb:
            print(action_space[idx], end=', ')
        print()


def recursive():
    import timeit
    env = Pyenv()
    st = StatCounter()
    for i in range(1):
        env.reset()
        env.prepare()
        # print(env.get_handcards())
        cards = env.get_handcards()[:15]
        cards = ['J', '10', '10', '7', '7', '6']

        # last_cards = ['3', '3']
        mask = get_mask_onehot60(cards, action_space, None).reshape(len(action_space), 15, 4).sum(-1).astype(np.uint8)
        valid = mask.sum(-1) > 0
        cards_target = Card.char2onehot60(cards).reshape(-1, 4).sum(-1).astype(np.uint8)
        t1 = timeit.default_timer()
        print(cards_target)
        print(mask[valid])
        combs = get_combinations_recursive(mask[valid, :], cards_target)
        print(combs)
        idx_mapping = dict(zip(range(valid.shape[0]), np.where(valid)[0]))

        # idx_must_be_contained = set(
        #     [idx for idx in range(1, 9085) if valid[idx] and CardGroup.to_cardgroup(action_space[idx]). \
        #         bigger_than(CardGroup.to_cardgroup(last_cards))])
        # print(idx_must_be_contained)
        combs = [[idx_mapping[idx] for idx in comb] for comb in combs]
        # combs = [comb for comb in combs if not idx_must_be_contained.isdisjoint(comb)]
        # fine_mask = np.zeros([len(combs), 21])
        # for i in range(len(combs)):
        #     for j in range(len(combs[i])):
        #         if combs[i][j] in idx_must_be_contained:
        #             fine_mask[i][j] = 1
        # print(fine_mask)
        t2 = timeit.default_timer()
        st.feed(t2 - t1)
        print(len(combs))
        for comb in combs:
            for idx in comb:
                print(action_space[idx], end=', ')
            print()
    print(st.average)



if __name__ == '__main__':
    # print(action_space[16:60])
    # dancing_link()
    recursive()
