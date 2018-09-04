from tensorpack import *
import os, sys

if os.name == 'nt':
    sys.path.insert(0, '../../build/Release')
else:
    sys.path.insert(0, '../../build.linux')
from env import get_combinations_nosplit, get_combinations_recursive
from logger import Logger
from utils import to_char
from card import Card, action_space, action_space_onehot60, Category, CardGroup, augment_action_space_onehot60, augment_action_space, clamp_action_idx
import numpy as np
import tensorflow as tf
from utils import get_mask, get_minor_cards, train_fake_action_60, get_masks, test_fake_action
from utils import get_seq_length, pick_minor_targets, to_char, to_value, get_mask_alter, get_mask_onehot60


class Predictor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.num_actions = [100, 21]
        self.encoding = np.load('../AutoEncoder/encoding.npy')
        print('predictor loaded')

    def pad_state(self, state):
        # since out net uses max operation, we just dup the last row and keep the result same
        newstates = []
        for s in state:
            assert s.shape[0] <= self.num_actions[1]
            s = np.concatenate([s, np.repeat(s[-1:, :], self.num_actions[1] - s.shape[0], axis=0)], axis=0)
            newstates.append(s)
        newstates = np.stack(newstates, axis=0)
        if len(state) < self.num_actions[0]:
            state = np.concatenate([newstates, np.repeat(newstates[-1:, :, :], self.num_actions[0] - newstates.shape[0], axis=0)], axis=0)
        else:
            state = newstates
        return state

    def pad_fine_mask(self, mask):
        if mask.shape[0] < self.num_actions[0]:
            mask = np.concatenate([mask, np.repeat(mask[-1:], self.num_actions[0] - mask.shape[0], 0)], 0)
        return mask

    def pad_action_space(self, available_actions):
        # print(available_actions)
        for i in range(len(available_actions)):
            available_actions[i] += [available_actions[i][-1]] * (self.num_actions[1] - len(available_actions[i]))
        if len(available_actions) < self.num_actions[0]:
            available_actions.extend([available_actions[-1]] * (self.num_actions[0] - len(available_actions)))

    def get_combinations(self, curr_cards_char, last_cards_char):
        if len(curr_cards_char) > 10:
            card_mask = Card.char2onehot60(curr_cards_char).astype(np.uint8)
            mask = augment_action_space_onehot60
            a = np.expand_dims(1 - card_mask, 0) * mask
            invalid_row_idx = set(np.where(a > 0)[0])
            if len(last_cards_char) == 0:
                invalid_row_idx.add(0)

            valid_row_idx = [i for i in range(len(augment_action_space)) if i not in invalid_row_idx]

            mask = mask[valid_row_idx, :]
            idx_mapping = dict(zip(range(mask.shape[0]), valid_row_idx))

            # augment mask
            # TODO: known issue: 555444666 will not decompose into 5554 and 66644
            combs = get_combinations_nosplit(mask, card_mask)
            combs = [([] if len(last_cards_char) == 0 else [0]) + [clamp_action_idx(idx_mapping[idx]) for idx in comb] for comb in combs]

            if len(last_cards_char) > 0:
                idx_must_be_contained = set(
                    [idx for idx in valid_row_idx if CardGroup.to_cardgroup(augment_action_space[idx]). \
                        bigger_than(CardGroup.to_cardgroup(last_cards_char))])
                combs = [comb for comb in combs if not idx_must_be_contained.isdisjoint(comb)]
                fine_mask = np.zeros([len(combs), self.num_actions[1]], dtype=np.bool)
                for i in range(len(combs)):
                    for j in range(len(combs[i])):
                        if combs[i][j] in idx_must_be_contained:
                            fine_mask[i][j] = True
            else:
                fine_mask = None
        else:
            mask = get_mask_onehot60(curr_cards_char, action_space, None).reshape(len(action_space), 15, 4).sum(-1).astype(
                np.uint8)
            valid = mask.sum(-1) > 0
            cards_target = Card.char2onehot60(curr_cards_char).reshape(-1, 4).sum(-1).astype(np.uint8)
            # do not feed empty to C++, which will cause infinite loop
            combs = get_combinations_recursive(mask[valid, :], cards_target)
            idx_mapping = dict(zip(range(valid.shape[0]), np.where(valid)[0]))

            combs = [([] if len(last_cards_char) == 0 else [0]) + [idx_mapping[idx] for idx in comb] for comb in combs]

            if len(last_cards_char) > 0:
                valid[0] = True
                idx_must_be_contained = set(
                    [idx for idx in range(len(action_space)) if valid[idx] and CardGroup.to_cardgroup(action_space[idx]). \
                        bigger_than(CardGroup.to_cardgroup(last_cards_char))])
                combs = [comb for comb in combs if not idx_must_be_contained.isdisjoint(comb)]
                fine_mask = np.zeros([len(combs), self.num_actions[1]], dtype=np.bool)
                for i in range(len(combs)):
                    for j in range(len(combs[i])):
                        if combs[i][j] in idx_must_be_contained:
                            fine_mask[i][j] = True
            else:
                fine_mask = None
        return combs, fine_mask

    def subsample_combs_masks(self, combs, masks, num_sample):
        if masks is not None:
            assert len(combs) == masks.shape[0]
        idx = np.random.permutation(len(combs))[:num_sample]
        return [combs[i] for i in idx], (masks[idx] if masks is not None else None)

    def get_state_and_action_space(self, is_comb, curr_cards_char=None, last_cards_char=None, prob_state=None, cand_state=None, cand_actions=None, action=None, fine_mask=None):
        if is_comb:
            combs, fine_mask = self.get_combinations(curr_cards_char, last_cards_char)
            if len(combs) > self.num_actions[0]:
                combs, fine_mask = self.subsample_combs_masks(combs, fine_mask, self.num_actions[0])
            # TODO: utilize temporal relations to speedup
            available_actions = [[action_space[idx] for idx in comb] for
                                 comb in combs]
            # if fine_mask is not None:
            #     fine_mask = np.concatenate([np.ones([fine_mask.shape[0], 1], dtype=np.bool), fine_mask[:, :20]], axis=1)
            assert len(combs) > 0
            # if len(combs) == 0:
            #     available_actions = [[[]]]
            #     fine_mask = np.zeros([1, num_actions[1]], dtype=np.bool)
            #     fine_mask[0, 0] = True
            if fine_mask is not None:
                fine_mask = self.pad_fine_mask(fine_mask)
            self.pad_action_space(available_actions)
            # if len(combs) < num_actions[0]:
            #     available_actions.extend([available_actions[-1]] * (num_actions[0] - len(combs)))
            state = [np.stack([self.encoding[idx] for idx in comb]) for comb in combs]
            assert len(state) > 0
            # if len(state) == 0:
            #     assert len(combs) == 0
            #     state = [np.array([encoding[0]])]
            test = action_space_onehot60 == Card.char2onehot60(last_cards_char)
            test = np.all(test, axis=1)
            target = np.where(test)[0]
            assert target.size == 1
            extra_state = np.concatenate([self.encoding[target[0]], prob_state])
            for i in range(len(state)):
                state[i] = np.concatenate([state[i], np.tile(extra_state[None, :], [state[i].shape[0], 1])], axis=-1)
            state = self.pad_state(state)
            assert state.shape[0] == self.num_actions[0] and state.shape[1] == self.num_actions[1]
        else:
            assert action is not None
            if fine_mask is not None:
                fine_mask = fine_mask[action]
            available_actions = cand_actions[action]
            state = cand_state[action:action + 1, :, :]
            state = np.repeat(state, self.num_actions[0], axis=0)
            assert state.shape[0] == self.num_actions[0] and state.shape[1] == self.num_actions[1]
        return state, available_actions, fine_mask

    def predict(self, handcards, last_cards, prob_state):
        # print('%s current cards' % ('lord' if role_id == 2 else 'farmer'), curr_cards_char)
        fine_mask_input = np.ones([max(self.num_actions[0], self.num_actions[1])], dtype=np.bool)
        # first hierarchy
        # print(handcards, last_cards)
        state, available_actions, fine_mask = self.get_state_and_action_space(True, curr_cards_char=handcards, last_cards_char=last_cards, prob_state=prob_state)
        # print(available_actions)
        q_values = self.predictor([state[None, :, :, :], np.array([True]), np.array([fine_mask_input])])[0][0]
        action = np.argmax(q_values)
        assert action < self.num_actions[0]
        # clamp action to valid range
        action = min(action, self.num_actions[0] - 1)

        # second hierarchy
        state, available_actions, fine_mask = self.get_state_and_action_space(False, cand_state=state, cand_actions=available_actions, action=action, fine_mask=fine_mask)
        if fine_mask is not None:
            fine_mask_input = fine_mask if fine_mask.shape[0] == max(self.num_actions[0], self.num_actions[1]) \
                else np.pad(fine_mask, (0, max(self.num_actions[0], self.num_actions[1]) - fine_mask.shape[0]), 'constant',
                            constant_values=(0, 0))
        q_values = self.predictor([state[None, :, :, :], np.array([False]), np.array([fine_mask_input])])[0][0]
        if fine_mask is not None:
            q_values = q_values[:self.num_actions[1]]
            assert np.all(q_values[np.where(np.logical_not(fine_mask))[0]] < -100)
            q_values[np.where(np.logical_not(fine_mask))[0]] = np.nan
        action = np.nanargmax(q_values)
        assert action < self.num_actions[1]
        # clamp action to valid range
        action = min(action, self.num_actions[1] - 1)
        intention = available_actions[action]
        return intention