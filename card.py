from collections import Counter
import numpy as np
import itertools
import functools


def get_action_space():
    actions = [[]]
    # max_cards = 20
    # single
    for card in Card.cards:
        actions.append([card])
    # pair
    for card in Card.cards:
        if card != '*' and card != '$':
            actions.append([card] * 2)
    # triple
    for card in Card.cards:
        if card != '*' and card != '$':
            actions.append([card] * 3)
    # 3 + 1
    for main in Card.cards:
        if main != '*' and main != '$':
            for extra in Card.cards:
                if extra != main:
                    actions.append([main] * 3 + [extra])
    # 3 + 2
    for main in Card.cards:
        if main != '*' and main != '$':
            for extra in Card.cards:
                if extra != main and extra != '*' and extra != '$':
                    actions.append([main] * 3 + [extra] * 2)
    # single sequence
    for start_v in range(Card.to_value('3'), Card.to_value('2')):
        for end_v in range(start_v + 5, Card.to_value('*')):
            seq = range(start_v, end_v)
            actions.append(Card.to_cards(seq))
    # double sequence
    for start_v in range(Card.to_value('3'), Card.to_value('2')):
        for end_v in range(start_v + 3, int(min(start_v + 20 / 2, Card.to_value('*')))):
            seq = range(start_v, end_v)
            actions.append(Card.to_cards(seq) * 2)
    # triple sequence
    for start_v in range(Card.to_value('3'), Card.to_value('2')):
        for end_v in range(start_v + 2, int(min(start_v + 20 / 3, Card.to_value('*')))):
            seq = range(start_v, end_v)
            actions.append(Card.to_cards(seq) * 3)
    # 3 + 1 sequence
    for start_v in range(Card.to_value('3'), Card.to_value('2')):
        for end_v in range(start_v + 2, int(min(start_v + 20 / 4, Card.to_value('*')))):
            seq = range(start_v, end_v)
            main = Card.to_cards(seq)
            remains = [card for card in Card.cards if card not in main]
            for extra in list(itertools.combinations(remains, end_v - start_v)):
                if not ('*' in list(extra) and '$' in list(extra)):
                    actions.append(main * 3 + list(extra))
    # 3 + 2 sequence
    for start_v in range(Card.to_value('3'), Card.to_value('2')):
        for end_v in range(start_v + 2, int(min(start_v + 20 / 5, Card.to_value('*')))):
            seq = range(start_v, end_v)
            main = Card.to_cards(seq)
            remains = [card for card in Card.cards if card not in main and card not in ['*', '$']]
            for extra in list(itertools.combinations(remains, end_v - start_v)):
                actions.append(main * 3 + list(extra) * 2)
    # bomb
    for card in Card.cards:
        if card != '*' and card != '$':
            actions.append([card] * 4)
    # bigbang
    actions.append(['*', '$'])
    # 4 + 1 + 1
    for main in Card.cards:
        if main != '*' and main != '$':
            remains = [card for card in Card.cards if card != main]
            for extra in list(itertools.combinations(remains, 2)):
                if not ('*' in list(extra) and '$' in list(extra)):
                    actions.append([main] * 4 + list(extra))

    for a in actions:
        a.sort(key=lambda c: Card.cards.index(c))
    return actions


class Card:
    cards = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', '*', '$']
    np_cards = np.array(cards)
    # full_cards = [x for pair in zip(cards, cards, cards, cards) for x in pair if x not in ['*', '$']]
    # full_cards += ['*', '$']
    cards.index('3')
    cards_to_onehot_idx = dict((x, i * 4) for (i, x) in enumerate(cards))
    cards_to_onehot_idx['*'] = 52
    cards_to_onehot_idx['$'] = 53
    cards_to_value = dict(zip(cards, range(len(cards))))
    value_to_cards = dict((v, c) for (c, v) in cards_to_value.items())

    def __init__(self):
        pass

    @staticmethod
    def to_onehot(cards):
        counts = Counter(cards)
        onehot = np.zeros(54)
        for x in cards:
            if x in ['*', '$']:
                onehot[Card.cards_to_onehot_idx[x]] = 1
            else:
                subvec = np.zeros(4)
                subvec[:counts[x]] = 1
                onehot[Card.cards_to_onehot_idx[x]:Card.cards_to_onehot_idx[x]+4] = subvec
        return onehot

    # convert char to 0-56 color cards
    @staticmethod
    def char2color(cards):
        result = np.zeros([len(cards)])
        mask = np.zeros([57])
        for i in range(len(cards)):
            ind = Card.cards.index(cards[i]) * 4
            while mask[ind] == 1:
                ind += 1
            mask[ind] = 1
            result[i] = ind
            
        return result

    @staticmethod
    def onehot2color(cards):
        result = []
        for i in range(len(cards)):
            if cards[i] == 0:
                continue
            if i == 53:
                result.append(56)
            else:
                result.append(i)
        return np.array(result)

    @staticmethod
    def to_value(card):
        if type(card) is list or type(card) is range:
            val = 0
            for c in card:
                val += Card.cards_to_value[c]
            return val
        else:
            return Card.cards_to_value[card]

    @staticmethod
    def to_cards(values):
        if type(values) is list or type(values) is range:
            cards = []
            for v in values:
                cards.append(Card.value_to_cards[v])
            return cards
        else:
            return Card.value_to_cards[values]
    
    @staticmethod
    def to_cards_from_3_17(values):
        return Card.np_cards[values-3].tolist()


class CardGroup:
    def __init__(self, cards, t, val):
        self.type = t
        self.cards = cards
        self.value = val

    def __len__(self):
        return len(self.cards)

    def bigger_than(self, g):
        if g.type == 'bigbang':
            return False
        if g.type == 'bomb':
            if (self.type == 'bomb' and self.value > g.value) or self.type == 'bigbang':
                return True
            else:
                return False
        if (self.type == 'bomb' or self.type == 'bigbang') or \
                (self.type == g.type and len(self) == len(g) and self.value > g.value):
            return True
        else:
            return False

    @staticmethod
    def isvalid(cards):
        return CardGroup.folks(cards) == 1

    @staticmethod
    def to_cardgroup(cards):
        candidates = CardGroup.analyze(cards)
        for c in candidates:
            if len(c.cards) == len(cards):
                return c
        print("cards error!")
        print(cards)
        raise Exception("Invalid Cards!")

    @staticmethod
    def folks(cards):
        cand = CardGroup.analyze(cards)
        cnt = 10000
        # if not cards:
        #     return 0
        # for c in cand:
        #     remain = list(cards)
        #     for card in c.cards:
        #         remain.remove(card)
        #     if CardGroup.folks(remain) + 1 < cnt:
        #         cnt = CardGroup.folks(remain) + 1
        # return cnt
        spec = False
        for c in cand:
            if c.type == 'triple_seq' or c.type == 'triple+single' or \
                    c.type == 'triple+double' or c.type == 'quadric+singles' or \
                    c.type == 'quadric+doubles' or c.type == 'triple_seq+singles' or \
                    c.type == 'triple_seq+doubles' or c.type == 'single_seq' or \
                    c.type == 'double_seq':
                spec = True
                remain = list(cards)
                for card in c.cards:
                    remain.remove(card)
                if CardGroup.folks(remain) + 1 < cnt:
                    cnt = CardGroup.folks(remain) + 1
        if not spec:
            cnt = len(cand)
        return cnt

    @staticmethod
    def analyze(cards):
        cards = list(cards)
        candidates = []

        counts = Counter(cards)
        if '*' in cards and '$' in cards:
            candidates.append((CardGroup(['*', '$'], 'bigbang', 10000)))
            cards.remove('*')
            cards.remove('$')

        quadrics = []
        # quadric
        for c in counts:
            if counts[c] == 4:
                quadrics.append(c)
                candidates.append(CardGroup([c] * 4, 'bomb', Card.to_value(c)))
                cards = list(filter(lambda a: a != c, cards))

        counts = Counter(cards)
        singles = [c for c in counts if counts[c] == 1]
        doubles = [c for c in counts if counts[c] == 2]
        triples = [c for c in counts if counts[c] == 3]

        singles.sort(key=lambda k: Card.cards_to_value[k])
        doubles.sort(key=lambda k: Card.cards_to_value[k])
        triples.sort(key=lambda k: Card.cards_to_value[k])

        # continuous sequence
        if len(singles) > 0:
            cnt = 1
            cand = [singles[0]]
            for i in range(1, len(singles)):
                if Card.to_value(singles[i]) >= Card.to_value('2'):
                    break
                if Card.to_value(singles[i]) == Card.to_value(cand[-1]) + 1:
                    cand.append(singles[i])
                    cnt += 1
                else:
                    if cnt >= 5:
                        candidates.append(CardGroup(cand, 'single_seq', Card.to_value(cand[-1])))
                        # for c in cand:
                        #     cards.remove(c)
                    cand = [singles[i]]
                    cnt = 1
            if cnt >= 5:
                candidates.append(CardGroup(cand, 'single_seq', Card.to_value(cand[-1])))
                # for c in cand:
                #     cards.remove(c)

        if len(doubles) > 0:
            cnt = 1
            cand = [doubles[0]] * 2
            for i in range(1, len(doubles)):
                if Card.to_value(doubles[i]) >= Card.to_value('2'):
                    break
                if Card.to_value(doubles[i]) == Card.to_value(cand[-1]) + 1:
                    cand += [doubles[i]] * 2
                    cnt += 1
                else:
                    if cnt >= 3:
                        candidates.append(CardGroup(cand, 'double_seq', Card.to_value(cand[-1])))
                        # for c in cand:
                            # if c in cards:
                            #     cards.remove(c)
                    cand = [doubles[i]] * 2
                    cnt = 1
            if cnt >= 3:
                candidates.append(CardGroup(cand, 'double_seq', Card.to_value(cand[-1])))
                # for c in cand:
                    # if c in cards:
                    #     cards.remove(c)

        if len(triples) > 0:
            cnt = 1
            cand = [triples[0]] * 3
            for i in range(1, len(triples)):
                if Card.to_value(triples[i]) >= Card.to_value('2'):
                    break
                if Card.to_value(triples[i]) == Card.to_value(cand[-1]) + 1:
                    cand += [triples[i]] * 3
                    cnt += 1
                else:
                    if cnt >= 2:
                        candidates.append(CardGroup(cand, 'triple_seq', Card.to_value(cand[-1])))
                        # for c in cand:
                        #     if c in cards:
                        #         cards.remove(c)
                    cand = [triples[i]] * 3
                    cnt = 1
            if cnt >= 2:
                candidates.append(CardGroup(cand, 'triple_seq', Card.to_value(cand[-1])))
                # for c in cand:
                #     if c in cards:
                #         cards.remove(c)

        for t in triples:
            candidates.append(CardGroup([t] * 3, 'triple', Card.to_value(t)))

        counts = Counter(cards)
        singles = [c for c in counts if counts[c] == 1]
        doubles = [c for c in counts if counts[c] == 2]

        # single
        for s in singles:
            candidates.append(CardGroup([s], 'single', Card.to_value(s)))

        # double
        for d in doubles:
            candidates.append(CardGroup([d] * 2, 'double', Card.to_value(d)))

        # 3 + 1, 3 + 2
        for c in triples:
            triple = [c] * 3
            for s in singles:
                if s not in triple:
                    candidates.append(CardGroup(triple + [s], 'triple+single',
                                                Card.to_value(c) * 1000 + Card.to_value(s)))
            for d in doubles:
                if d not in triple:
                    candidates.append(CardGroup(triple + [d] * 2, 'triple+double',
                                                Card.to_value(c) * 1000 + Card.to_value(d)))

        # 4 + 2
        for c in quadrics:
            for extra in list(itertools.combinations(singles, 2)):
                candidates.append(CardGroup([c] * 4 + list(extra), 'quadric+singles',
                                            Card.to_value(c) * 1000 + Card.to_value(list(extra))))
            for extra in list(itertools.combinations(doubles, 2)):
                candidates.append(CardGroup([c] * 4 + list(extra) * 2, 'quadric+doubles',
                                            Card.to_value(c) * 1000 + Card.to_value(list(extra))))
        # 3 * n + n, 3 * n + 2 * n
        triple_seq = [c.cards for c in candidates if c.type == 'triple_seq']
        for cand in triple_seq:
            cnt = int(len(cand) / 3)
            for extra in list(itertools.combinations(singles, cnt)):
                candidates.append(
                    CardGroup(cand + list(extra), 'triple_seq+singles',
                              Card.to_value(cand[-1]) * 1000 + Card.to_value(list(extra))))
            for extra in list(itertools.combinations(doubles, cnt)):
                candidates.append(
                    CardGroup(cand + list(extra) * 2, 'triple_seq+doubles',
                              Card.to_value(cand[-1]) * 1000 + Card.to_value(list(extra))))

        importance = ['single', 'double', 'double_seq', 'single_seq', 'triple+single',
                      'triple+double', 'triple_seq+singles', 'triple_seq+doubles',
                      'triple_seq', 'triple', 'quadric+singles', 'quadric+doubles',
                      'bomb', 'bigbang']
        candidates.sort(key=functools.cmp_to_key(lambda x, y: importance.index(x.type) - importance.index(y.type)
                        if importance.index(x.type) != importance.index(y.type) else x.value - y.value))
        # for c in candidates:
        #     print c.cards
        return candidates

action_space = get_action_space()

if __name__ == '__main__':
    pass
    # CardGroup.to_cardgroup(['6', '6', 'Q', 'Q', 'Q'])
    actions = get_action_space()
    for i in range(1, len(actions)):
        CardGroup.to_cardgroup(actions[i])
    # print(CardGroup.folks(['3', '4', '3', '4', '3', '4', '*', '$']))
    # CardGroup.to_cardgroup(['3', '4', '3', '4', '3', '4', '*', '$'])
    # print actions[561]
    # print CardGroup.folks(actions[561])
    # CardGroup.to_cardgroup(actions[i])
    # print Card.to_onehot(['3', '4', '4', '$'])
    # print len(actions)
    # print Card.to_cards(1)
    # CardGroup.analyze(['3', '3', '3', '4', '4', '4', '10', 'J', 'Q', 'A', 'A', '2', '2', '*', '$'])