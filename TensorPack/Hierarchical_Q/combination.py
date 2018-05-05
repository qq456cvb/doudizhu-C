import numpy as np
from pyenv import Pyenv
import os
import sys
if os.name == 'nt':
    sys.path.insert(0, '../../build/Release')
else:
    sys.path.insert(0, '../../build.linux')
from env import get_combinations
from utils import get_mask_onehot60, get_mask
from card import action_space, Card, action_space_category, Category


def dancing_link():
    env = Pyenv()
    env.reset()
    env.prepare()
    # print(env.get_handcards())
    # cards = env.get_handcards()
    cards = ['3', '3', '3']
    mask = get_mask_onehot60(cards, action_space, None).astype(np.uint8)

    # augment mask
    # TODO: known issue: 555444666 will not decompose into 5554 and 66644
    augmented_action_space = action_space.copy()
    singles = []
    single_mask = get_mask(cards, action_space_category[Category.SINGLE])
    for i in range(single_mask.size - 2):
        if single_mask[i] > 0:
            augmented_action_space += [action_space_category[Category.SINGLE][i]] * 3
            for j in range(1, 4):
                tmp = np.zeros([60])
                tmp[i * 4 + j] = 1
                singles.append(tmp)

    doubles = []
    double_mask = get_mask(cards, action_space_category[Category.DOUBLE])
    for i in range(double_mask.size):
        if double_mask[i] > 0:
            augmented_action_space += [action_space_category[Category.DOUBLE][i]]
            tmp = np.zeros([60])
            tmp[i * 4 + 2:i * 4 + 4] = 1
            doubles.append(tmp)

    mask = np.concatenate([mask, np.stack(singles + doubles)])
    # print(Card.char2onehot60(cards).astype(np.uint8))
    combs = get_combinations(mask, Card.char2onehot60(cards).astype(np.uint8))
    print(len(combs))
    for comb in combs:
        for idx in comb:
            print(augmented_action_space[idx], end=', ')
        print()


def recursive():
    env = Pyenv()
    env.reset()
    env.prepare()
    # print(env.get_handcards())
    cards = env.get_handcards()[:20]
    # cards = ['3', '3',  '4', '4']
    mask = get_mask_onehot60(cards, action_space, None).reshape(len(action_space), 15, 4).sum(-1).astype(np.uint8)
    valid = mask.sum(-1) > 0
    cards_target = Card.char2onehot60(cards).reshape(-1, 4).sum(-1).astype(np.uint8)
    combs = get_combinations(mask, cards_target, valid)
    print(len(combs))
    for comb in combs:
        for idx in comb:
            print(action_space[idx], end=', ')
        print()


if __name__ == '__main__':
    # print(action_space[16:60])
    recursive()