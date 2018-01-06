import card
from card import action_space, Category, action_space
import numpy as np
from collections import Counter
import tensorflow as tf

action_space_single = action_space[1:16]
action_space_pair = action_space[16:29]
action_space_triple = action_space[29:42]
action_space_quadric = action_space[42:55]

##################################################### UTILITIES ########################################################
def counter_subset(list1, list2):
    c1, c2 = Counter(list1), Counter(list2)

    for (k, n) in c1.items():
        if n > c2[k]:
            return False
    return True

# map char cards to 3 - 17
def to_value(cards):
    values = [card.Card.cards.index(c)+3 for c in cards]
    return values

# map 3 - 17 to char cards
def to_char(cards):
    chars = [card.Card.cards[c-3] for c in cards]
    return chars

def get_mask(cards, action_space, last_cards):
    # 1 valid; 0 invalid
    mask = np.zeros([len(action_space)])
    for j in range(mask.size):
        if counter_subset(action_space[j], cards):
            mask[j] = 1
    mask = mask.astype(bool)
    if last_cards:
        for j in range(1, mask.size):
            if mask[j] == 1 and not card.CardGroup.to_cardgroup(action_space[j]).\
                    bigger_than(card.CardGroup.to_cardgroup(last_cards)):
                mask[j] = False
    # else:
    #     mask[0] = False
    return mask

# # get char cards, return valid response
# def get_mask_category(cards, action_space, last_cards=None):
#     mask = np.zeros([14]) if last_cards is None else np.zeros([15])
#     for i in range(action_space):
#         if counter_subset(action_space[i], cards):
#             if last_cards is None:
#                 mask[char2value_3_17(action_space[i][0])-3] = 1
#             else:
#                 diff = char2value_3_17(action_space[i][0]) - char2value_3_17(last_cards[0])
#                 if diff > 0:
#                     mask[diff-1] = 1
#     return mask.astype(bool)

def get_seq_length(category, cards_val):
    if category == Category.SINGLE_LINE.value:
        return cards_val.size
    if category == Category.DOUBLE_LINE.value:
        return cards_val.size // 2
    if category == Category.TRIPLE_LINE.value:
        return cards_val.size // 3
    if category == Category.THREE_ONE_LINE.value:
        return cards_val.size // 4
    if category == Category.THREE_TWO_LINE.value:
        return cards_val.size // 5
    return None

# get [-1, 1] minor cards target, input: value cards 3-17
def find_minor_in_three_one(cards):
    if cards[0] == cards[1]:
        return cards[-1]
    else:
        return cards[0]

def find_minor_in_three_two(cards):
    if cards[1] == cards[2]:
        return cards[-1]
    else:
        return cards[0]

def find_minor_in_three_one_line(cards):
    cnt = np.zeros([18])
    for i in range(len(cards)):
        cnt[cards[i]] += 1
    minor = []
    for i in range(3, 18):
        if cnt[i] == 1:
            minor.append(i)
    return np.array(minor)

def find_minor_in_three_two_line(cards):
    cnt = np.zeros([18])
    for i in range(len(cards)):
        cnt[cards[i]] += 1
    minor = []
    for i in range(3, 18):
        if cnt[i] == 2:
            minor.append(i)
    return np.array(minor)

def find_minor_in_four_two(cards):
    cnt = np.zeros([18])
    for i in range(len(cards)):
        cnt[cards[i]] += 1
    minor = []
    for i in range(3, 18):
        if cnt[i] == 1:
            minor.append(i)
    return np.array(minor)

def get_minor_cards(cards, category_idx):
    minor_cards = np.ones([15])
    length = 0
    if category_idx == Category.THREE_ONE.value:
        length = 1
        minor_cards[find_minor_in_three_one(cards)-3] = -1
    if category_idx == Category.THREE_TWO.value:
        length = 1
        minor_cards[find_minor_in_three_two(cards)-3] = -1
    if category_idx == Category.THREE_ONE_LINE.value:
        length = int(cards.size / 4)
        minor_cards[find_minor_in_three_one_line(cards)-3] = -1
    if category_idx == Category.THREE_TWO_LINE.value:
        length = int(cards.size / 5)
        minor_cards[find_minor_in_three_two_line(cards)-3] = -1
    if category_idx == Category.FOUR_TWO.value:
        length = 2
        minor_cards[find_minor_in_four_two(cards)-3] = -1
    return minor_cards, length


def discounted_return(r, gamma):
    r = r.astype(float)
    r_out = np.zeros_like(r)
    val = 0
    for i in reversed(range(r.shape[0])):
        r_out[i] = r[i] + gamma * val
        val = r_out[i]
    return r_out


def get_feature_state(env, mask=None):
    curr_cards = to_char(env.get_curr_handcards())
    curr_val, curr_round = env.get_cards_value(card.Card.char2color(curr_cards))
    if mask is None:
        mask = get_mask(curr_cards, action_space, to_char(env.get_last_outcards()))
    features = np.zeros([len(mask), 9])
    features[:, 0] = mask.astype(np.int32)
    for i in range(mask.shape[0]):
        m = mask[i]
        if m:
            a = action_space[i]
            
            if not a:
                features[i, 1] = 1
                continue
            next_cards = curr_cards.copy()
            for c in a:
                next_cards.remove(c)
            next_val, next_round = env.get_cards_value(card.Card.char2color(next_cards))
            lose_control = env.will_lose_control(card.Card.char2value_3_17(a) + 3)
            if lose_control:
                features[i, 1] = 1
            if len(a) >= len(curr_cards):
                features[i, 2] = 1
            if next_val > curr_val:
                features[i, 3] = 1
            if next_round < curr_round:
                features[i, 4] = 1

            cnt = len(a)
            if cnt > 15:
                cnt = 15
            features[i, 5] = cnt & 8 >> 3
            features[i, 6] = cnt & 4 >> 2
            features[i, 7] = cnt & 2 >> 1
            features[i, 8] = cnt & 1
    return features


def get_masks(handcards):
    input_single = get_mask(handcards, action_space_single, None)
    input_pair = get_mask(handcards, action_space_pair, None)
    input_triple = get_mask(handcards, action_space_triple, None)
    input_quadric = get_mask(handcards, action_space_quadric, None)
    return input_single, input_pair, input_triple, input_quadric


# receive targets and handcards as chars
def train_fake_action(targets, handcards, s, sess, network):
    acc = []
    for target in targets:
        target_val = card.Card.char2value_3_17(target) - 3
        input_single, input_pair, input_triple, input_quadric = get_masks(handcards)
        _, response_active_output, fake_loss = sess.run([network.optimize_fake, 
            network.fc_response_active_output, 
            network.active_response_loss],
                feed_dict = {
                    network.training: True,
                    network.input_state: s,
                    network.input_single: np.reshape(input_single, [1, -1]),
                    network.input_pair: np.reshape(input_pair, [1, -1]),
                    network.input_triple: np.reshape(input_triple, [1, -1]),
                    network.input_quadric: np.reshape(input_quadric, [1, -1]),
                    network.active_response_input: np.array([target_val]),
            })
        handcards.remove(target)
        acc.append(1 if np.argmax(response_active_output) == target_val else 0)
    return acc

def pick_minor_targets(category, cards_char):
    if category == Category.THREE_ONE.value:
        return [cards_char[-1]]
    if category == Category.THREE_TWO.value:
        return cards_char[-2:]
    if category == Category.THREE_ONE_LINE.value:
        length = len(cards_char) // 4
        return cards_char[-length:]
    if category == Category.THREE_TWO_LINE.value:
        length = len(cards_char) // 5
        return cards_char[-length*2:]
    if category == Category.FOUR_TWO.value:
        return cards_char[-2:]
    return None
    
    