import numpy as np
from pyenv import Pyenv
import os
import sys
if os.name == 'nt':
    sys.path.insert(0, '../../build/Release')
else:
    sys.path.insert(0, '../../build.linux')
from env import get_combinations_recursive, get_combinations_nosplit
from utils import get_mask_onehot60, get_mask
from card import action_space, Card, action_space_category, Category, augment_action_space_onehot60, CardGroup, augment_action_space
from tensorpack.utils.stats import StatCounter


def dancing_link():
    env = Pyenv()
    env.reset()
    env.prepare()
    # print(env.get_handcards())
    cards = env.get_handcards()
    import timeit
    begin = timeit.default_timer()
    card_mask = Card.char2onehot60(cards).astype(np.uint8)
    # mask = get_mask_onehot60(cards, action_space, None).astype(np.uint8)
    # last_cards = ['3', '3']
    mask = augment_action_space_onehot60
    a = np.expand_dims(1 - card_mask, 0) * mask
    row_idx = np.where(a > 0)[0]

    tmp = np.ones(len(augment_action_space))
    tmp[row_idx] = 0
    valid_row_idx = np.where(tmp > 0)[0]
    # valid_row_idx = [idx for idx in valid_row_idx if CardGroup.to_cardgroup(action_space[idx]).\
    #                 bigger_than(CardGroup.to_cardgroup(last_cards))]

    mask = mask[valid_row_idx, :]
    idx_mapping = dict(zip(range(mask.shape[0]), valid_row_idx))

    # augment mask
    # TODO: known issue: 555444666 will not decompose into 5554 and 66644

    combs = get_combinations_nosplit(mask, Card.char2onehot60(cards).astype(np.uint8))
    end = timeit.default_timer()
    print(end - begin)
    print(len(combs))
    # for comb in combs:
    #     for idx in comb:
    #         print(augment_action_space[idx_mapping[idx]], end=', ')
    #     print()


def recursive():
    import timeit
    env = Pyenv()
    st = StatCounter()
    for i in range(100):
        env.reset()
        env.prepare()
        # print(env.get_handcards())
        cards = env.get_handcards()[:12]
        # cards = ['3', '3',  '4', '4']
        mask = get_mask_onehot60(cards, action_space, None).reshape(len(action_space), 15, 4).sum(-1).astype(np.uint8)
        valid = mask.sum(-1) > 0
        cards_target = Card.char2onehot60(cards).reshape(-1, 4).sum(-1).astype(np.uint8)
        t1 = timeit.default_timer()
        combs = get_combinations_recursive(mask, cards_target, valid)
        t2 = timeit.default_timer()
        st.feed(t2 - t1)
        # print(len(combs))
    print(st.average)
    # for comb in combs:
    #     for idx in comb:
    #         print(action_space[idx], end=', ')
    #     print()


if __name__ == '__main__':
    # print(action_space[16:60])
    # dancing_link()
    dancing_link()
