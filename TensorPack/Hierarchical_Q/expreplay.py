#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: expreplay.py
# Adapted by: Neil You for Fight the Lord
import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

import numpy as np
import copy
from collections import deque, namedtuple
import threading
from six.moves import queue, range

from tensorpack.dataflow import DataFlow
from tensorpack.utils import logger
from tensorpack.utils.utils import get_tqdm, get_rng
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.concurrency import LoopThread, ShareSessionThread
from tensorpack.callbacks.base import Callback
from card import action_space, action_space_onehot60, Card, CardGroup, augment_action_space_onehot60, clamp_action_idx, augment_action_space
from utils import to_char, to_value, get_mask_onehot60
from env import Env as CEnv
from env import get_combinations_nosplit, get_combinations_recursive

import random


ROLE_ID_TO_TRAIN = 2

__all__ = ['ExpReplay']

Experience = namedtuple('Experience',
                        ['joint_state', 'action', 'reward', 'isOver', 'comb_mask', 'fine_mask'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape):
        self.max_size = int(max_size)
        self.state_shape = state_shape

        self.state = np.zeros((self.max_size,) + state_shape, dtype='float32')
        self.action = np.zeros((self.max_size,), dtype='int32')
        self.reward = np.zeros((self.max_size,), dtype='float32')
        self.isOver = np.zeros((self.max_size,), dtype='bool')
        self.comb_mask = np.zeros((self.max_size,), dtype='bool')
        self.fine_mask = np.zeros((self.max_size, max(state_shape[0], state_shape[1])), dtype='bool')

        self._curr_size = 0
        self._curr_pos = 0

    def append(self, exp):
        """
        Args:
            exp (Experience):
        """
        if self._curr_size < self.max_size:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
            self._curr_size += 1
        else:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size

    def sample(self, idx):
        """ return a tuple of (s,r,a,o),
            where s is of shape STATE_SIZE + (2,)"""
        idx = (self._curr_pos + idx) % self._curr_size
        action = self.action[idx]
        reward = self.reward[idx]
        isOver = self.isOver[idx]
        comb_mask = self.comb_mask[idx]
        if idx + 2 <= self._curr_size:
            state = self.state[idx:idx+2]
            fine_mask = self.fine_mask[idx:idx+2]
        else:
            end = idx + 2 - self._curr_size
            state = self._slice(self.state, idx, end)
            fine_mask = self._slice(self.fine_mask, idx, end)
        return state, action, reward, isOver, comb_mask, fine_mask

    def _slice(self, arr, start, end):
        s1 = arr[start:]
        s2 = arr[:end]
        return np.concatenate((s1, s2), axis=0)

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        self.state[pos] = exp.joint_state
        self.action[pos] = exp.action
        self.reward[pos] = exp.reward
        self.isOver[pos] = exp.isOver
        self.comb_mask[pos] = exp.comb_mask
        self.fine_mask[pos] = exp.fine_mask


class ExpReplay(DataFlow, Callback):
    """
    Implement experience replay in the paper
    `Human-level control through deep reinforcement learning
    <http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html>`_.
    This implementation provides the interface as a :class:`DataFlow`.
    This DataFlow is __not__ fork-safe (thus doesn't support multiprocess prefetching).
    This implementation assumes that state is
    batch-able, and the network takes batched inputs.
    """

    def __init__(self,
                 predictor_io_names,
                 player,
                 state_shape,
                 num_actions,
                 batch_size,
                 memory_size, init_memory_size,
                 init_exploration,
                 update_frequency,
                 encoding_file='../AutoEncoder/encoding.npy'):
        """
        Args:
            predictor_io_names (tuple of list of str): input/output names to
                predict Q value from state.
            player (RLEnvironment): the player.
            history_len (int): length of history frames to concat. Zero-filled
                initial frames.
            update_frequency (int): number of new transitions to add to memory
                after sampling a batch of transitions for training.
        """
        init_memory_size = int(init_memory_size)

        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.exploration = init_exploration
        self.num_actions = num_actions
        self.encoding = np.load(encoding_file)
        logger.info("Number of Legal actions: {}, {}".format(*self.num_actions))

        self.rng = get_rng(self)
        self._init_memory_flag = threading.Event()  # tell if memory has been initialized

        # a queue to receive notifications to populate memory
        self._populate_job_queue = queue.Queue(maxsize=5)

        self.mem = ReplayMemory(memory_size, state_shape)
        self.player.reset()
        # init_cards = np.arange(36)
        # self.player.prepare_manual(init_cards)
        self.player.prepare()
        # self._current_ob = self.player.get_state_prob()
        self._comb_mask = True
        self._fine_mask = None
        self._current_ob, self._action_space = self.get_state_and_action_spaces()
        self._player_scores = StatCounter()
        self._current_game_score = StatCounter()

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
                self._fine_mask = np.zeros([len(combs), self.num_actions[1]], dtype=np.bool)
                for i in range(len(combs)):
                    for j in range(len(combs[i])):
                        if combs[i][j] in idx_must_be_contained:
                            self._fine_mask[i][j] = True
            else:
                self._fine_mask = None
        else:
            mask = get_mask_onehot60(curr_cards_char, action_space, None).reshape(len(action_space), 15, 4).sum(-1).astype(
                np.uint8)
            valid = mask.sum(-1) > 0
            cards_target = Card.char2onehot60(curr_cards_char).reshape(-1, 4).sum(-1).astype(np.uint8)
            combs = get_combinations_recursive(mask[valid, :], cards_target)
            idx_mapping = dict(zip(range(valid.shape[0]), np.where(valid)[0]))

            combs = [([] if len(last_cards_char) == 0 else [0]) + [idx_mapping[idx] for idx in comb] for comb in combs]

            if len(last_cards_char) > 0:
                valid[0] = True
                idx_must_be_contained = set(
                    [idx for idx in range(len(action_space)) if valid[idx] and CardGroup.to_cardgroup(action_space[idx]). \
                        bigger_than(CardGroup.to_cardgroup(last_cards_char))])
                combs = [comb for comb in combs if not idx_must_be_contained.isdisjoint(comb)]
                self._fine_mask = np.zeros([len(combs), self.num_actions[1]], dtype=np.bool)
                for i in range(len(combs)):
                    for j in range(len(combs[i])):
                        if combs[i][j] in idx_must_be_contained:
                            self._fine_mask[i][j] = True
            else:
                self._fine_mask = None
        return combs

    def subsample_combs_masks(self, combs, masks, num_sample):
        if masks is not None:
            assert len(combs) == masks.shape[0]
        idx = np.random.permutation(len(combs))[:num_sample]
        return [combs[i] for i in idx], (masks[idx] if masks is not None else None)

    def get_state_and_action_spaces(self, action=None):

        def cards_char2embedding(cards_char):
            test = (action_space_onehot60 == Card.char2onehot60(cards_char))
            test = np.all(test, axis=1)
            target = np.where(test)[0]
            return self.encoding[target[0]]

        last_two_cards_char = self.player.get_last_two_cards()
        last_two_cards_char = [to_char(cards) for cards in last_two_cards_char]
        last_cards_char = last_two_cards_char[0]
        if not last_cards_char:
            last_cards_char = last_two_cards_char[1]
        curr_cards_char = to_char(self.player.get_curr_handcards())
        if self._comb_mask:
            # print(curr_cards_char, last_cards_char)
            combs = self.get_combinations(curr_cards_char, last_cards_char)
            if len(combs) > self.num_actions[0]:
                combs, self._fine_mask = self.subsample_combs_masks(combs, self._fine_mask, self.num_actions[0])
            # TODO: utilize temporal relations to speedup
            available_actions = [[action_space[idx] for idx in comb] for comb in combs]
            # print(available_actions)
            # print('-------------------------------------------')
            assert len(combs) > 0
            if self._fine_mask is not None:
                self._fine_mask = self.pad_fine_mask(self._fine_mask)
            self.pad_action_space(available_actions)
            state = [np.stack([self.encoding[idx] for idx in comb]) for comb in combs]
            assert len(state) > 0
            prob_state = self.player.get_state_prob()
            # test = action_space_onehot60 == Card.char2onehot60(last_cards_char)
            # test = np.all(test, axis=1)
            # target = np.where(test)[0]
            # assert target.size == 1
            extra_state = np.concatenate([cards_char2embedding(last_two_cards_char[0]), cards_char2embedding(last_two_cards_char[1]), prob_state])
            for i in range(len(state)):
                state[i] = np.concatenate([state[i], np.tile(extra_state[None, :], [state[i].shape[0], 1])], axis=-1)
            state = self.pad_state(state)
            assert state.shape[0] == self.num_actions[0] and state.shape[1] == self.num_actions[1]
        else:
            assert action is not None
            if self._fine_mask is not None:
                self._fine_mask = self._fine_mask[action]
            available_actions = self._action_space[action]
            state = self._current_ob[action:action+1, :, :]
            state = np.repeat(state, self.num_actions[0], axis=0)
            assert state.shape[0] == self.num_actions[0] and state.shape[1] == self.num_actions[1]
        return state, available_actions

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

    # input is a list of N * HIDDEN_STATE
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

    def get_simulator_thread(self):
        # spawn a separate thread to run policy
        def populate_job_func():
            self._populate_job_queue.get()
            for _ in range(self.update_frequency):
                self._populate_exp()
        th = ShareSessionThread(LoopThread(populate_job_func, pausable=False))
        th.name = "SimulatorThread"
        return th

    def _init_memory(self):
        logger.info("Populating replay memory with epsilon={} ...".format(self.exploration))

        with get_tqdm(total=self.init_memory_size) as pbar:
            while len(self.mem) < self.init_memory_size:
                self._populate_exp()
                pbar.update()
        self._init_memory_flag.set()

    def _populate_exp(self):
        """ populate a transition by epsilon-greedy"""
        old_s = self._current_ob
        comb_mask = self._comb_mask
        if not self._comb_mask and self._fine_mask is not None:
            fine_mask = self._fine_mask if self._fine_mask.shape[0] == max(self.num_actions[0], self.num_actions[1]) \
                else np.pad(self._fine_mask, (0, max(self.num_actions[0], self.num_actions[1]) - self._fine_mask.shape[0]), 'constant', constant_values=(0, 0))
        else:
            fine_mask = np.ones([max(self.num_actions[0], self.num_actions[1])], dtype=np.bool)
        last_cards_value = self.player.get_last_outcards()
        if self.rng.rand() <= self.exploration:
            if not self._comb_mask and self._fine_mask is not None:
                q_values = np.random.rand(self.num_actions[1])
                q_values[np.where(np.logical_not(self._fine_mask))[0]] = np.nan
                act = np.nanargmax(q_values)
                # print(q_values)
                # print(act)
            else:
                act = self.rng.choice(range(self.num_actions[0 if comb_mask else 1]))
        else:
            q_values = self.predictor(old_s[None, :, :, :], np.array([comb_mask]), np.array([fine_mask]))[0][0]
            if not self._comb_mask and self._fine_mask is not None:
                q_values = q_values[:self.num_actions[1]]
                assert np.all(q_values[np.where(np.logical_not(self._fine_mask))[0]] < -100)
                q_values[np.where(np.logical_not(self._fine_mask))[0]] = np.nan
            act = np.nanargmax(q_values)
            assert act < self.num_actions[0 if comb_mask else 1]
            # print(q_values)
            # print(act)
            # clamp action to valid range
            act = min(act, self.num_actions[0 if comb_mask else 1] - 1)
        if comb_mask:
            reward = 0
            isOver = False
        else:
            if last_cards_value.size > 0:
                if act > 0:
                    if not CardGroup.to_cardgroup(self._action_space[act]).bigger_than(CardGroup.to_cardgroup(to_char(last_cards_value))):
                        print('warning, some error happened')
            # print(to_char(self.player.get_curr_handcards()))
            reward, isOver, _ = self.player.step_manual(to_value(self._action_space[act]))

            # print(self._action_space[act])

        # step for AI
        while not isOver and self.player.get_role_ID() != ROLE_ID_TO_TRAIN:
            _, reward, _ = self.player.step_auto()
            isOver = (reward != 0)
        # if landlord negate the reward
        if ROLE_ID_TO_TRAIN == 2:
            reward = -reward
        self._current_game_score.feed(reward)

        if isOver:
            # print('lord wins' if reward > 0 else 'farmer wins')
            self._player_scores.feed(self._current_game_score.sum)
            # print(self._current_game_score.sum)
            while True:
                self.player.reset()
                # init_cards = np.arange(36)
                # self.player.prepare_manual(init_cards)
                self.player.prepare()
                self._comb_mask = True
                early_stop = False
                while self.player.get_role_ID() != ROLE_ID_TO_TRAIN:
                    _, reward, _ = self.player.step_auto()
                    isOver = (reward != 0)
                    if isOver:
                        print('prestart ends too early! now resetting env')
                        early_stop = True
                        break
                if early_stop:
                    continue
                self._current_ob, self._action_space = self.get_state_and_action_spaces()
                break
            self._current_game_score.reset()
        else:
            self._comb_mask = not self._comb_mask
        self._current_ob, self._action_space = self.get_state_and_action_spaces(act if not self._comb_mask else None)
        self.mem.append(Experience(old_s, act, reward, isOver, comb_mask, fine_mask))

    def debug(self, cnt=100000):
        with get_tqdm(total=cnt) as pbar:
            for i in range(cnt):
                self.mem.append(Experience(np.zeros([self.num_actions[0], self.num_actions[1], 256]), 0, 0, False, True if i % 2 == 0 else False))
                # self._current_ob, self._action_space = self.get_state_and_action_spaces(None)
                pbar.update()

    def get_data(self):
        # wait for memory to be initialized
        self._init_memory_flag.wait()

        while True:
            idx = self.rng.randint(
                self._populate_job_queue.maxsize * self.update_frequency,
                len(self.mem) - 1,
                size=self.batch_size)
            batch_exp = [self.mem.sample(i) for i in idx]

            yield self._process_batch(batch_exp)
            self._populate_job_queue.put(1)

    def _process_batch(self, batch_exp):
        state = np.asarray([e[0] for e in batch_exp], dtype='float32')
        action = np.asarray([e[1] for e in batch_exp], dtype='int32')
        reward = np.asarray([e[2] for e in batch_exp], dtype='float32')
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        comb_mask = np.asarray([e[4] for e in batch_exp], dtype='bool')
        fine_mask = np.asarray([e[5] for e in batch_exp], dtype='bool')
        return [state, action, reward, isOver, comb_mask, fine_mask]

    def _setup_graph(self):
        self.predictor = self.trainer.get_predictor(*self.predictor_io_names)

    def _before_train(self):
        while self.player.get_role_ID() != ROLE_ID_TO_TRAIN:
            self.player.step_auto()
            self._current_ob, self._action_space = self.get_state_and_action_spaces()
        self._init_memory()
        self._simulator_th = self.get_simulator_thread()
        self._simulator_th.start()

    def _trigger(self):
        v = self._player_scores
        try:
            mean, max = v.average, v.max
            self.trainer.monitors.put_scalar('expreplay/mean_score', mean)
            self.trainer.monitors.put_scalar('expreplay/max_score', max)
        except Exception:
            logger.exception("Cannot log training scores.")
        v.reset()


if __name__ == '__main__':
    def predictor(x):
        return [np.random.random([1, 100])]
    player = CEnv()
    E = ExpReplay(
        predictor_io_names=(['state', 'comb_mask'], ['Qvalue']),
        player=CEnv(),
        state_shape=(100, 21, 256),
        num_actions=[100, 21],
        batch_size=16,
        memory_size=1e4,
        init_memory_size=1e4,
        init_exploration=0.,
        update_frequency=4
    )
    E.predictor = predictor
    E._init_memory()
    # for k in E.get_data():
    #     pass

    # for k in E.get_data():
    #     import IPython as IP
    #     IP.embed(config=IP.terminal.ipapp.load_default_config())
    #     pass
        # import IPython;
        # IPython.embed(config=IPython.terminal.ipapp.load_default_config())
        # break
