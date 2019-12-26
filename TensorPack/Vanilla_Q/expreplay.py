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
from utils import to_char, to_value, get_mask_onehot60, get_mask
from env import Env as CEnv
from env import get_combinations_nosplit, get_combinations_recursive

import random


ROLE_ID_TO_TRAIN = 2

__all__ = ['ExpReplay']

Experience = namedtuple('Experience',
                        ['joint_state', 'next_mask', 'action', 'reward', 'isOver'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape):
        self.max_size = int(max_size)
        self.state_shape = state_shape

        self.state = np.zeros((self.max_size,) + state_shape, dtype='float32')
        self.next_mask = np.zeros((self.max_size, 13527), dtype='bool')
        self.action = np.zeros((self.max_size,), dtype='int32')
        self.reward = np.zeros((self.max_size,), dtype='float32')
        self.isOver = np.zeros((self.max_size,), dtype='bool')

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
        if idx + 2 <= self._curr_size:
            next_mask = self.next_mask[idx + 1]
            state = self.state[idx:idx+2]
        else:
            next_mask = self.next_mask[0]
            end = idx + 2 - self._curr_size
            state = self._slice(self.state, idx, end)
        return state, next_mask, action, reward, isOver

    def _slice(self, arr, start, end):
        s1 = arr[start:]
        s2 = arr[:end]
        return np.concatenate((s1, s2), axis=0)

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        self.state[pos] = exp.joint_state
        self.next_mask[pos] = exp.next_mask
        self.action[pos] = exp.action
        self.reward[pos] = exp.reward
        self.isOver[pos] = exp.isOver


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
        logger.info("Number of Legal actions: {}".format(self.num_actions))

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
        self._current_ob = self.get_state()
        self._player_scores = StatCounter()
        self._current_game_score = StatCounter()

    def get_state(self):
        def cards_char2embedding(cards_char):
            test = (action_space_onehot60 == Card.char2onehot60(cards_char))
            test = np.all(test, axis=1)
            target = np.where(test)[0]
            return self.encoding[target[0]]
        s = self.player.get_state_prob()
        s = np.concatenate([Card.val2onehot60(self.player.get_curr_handcards()), s])
        last_two_cards_char = self.player.get_last_two_cards()
        last_two_cards_char = [to_char(c) for c in last_two_cards_char]
        return np.concatenate([s, cards_char2embedding(last_two_cards_char[0]), cards_char2embedding(last_two_cards_char[1])])

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
        mask = get_mask(to_char(self.player.get_curr_handcards()), action_space,
                        to_char(self.player.get_last_outcards()))
        if self.rng.rand() <= self.exploration:
            act = self.rng.choice(range(self.num_actions))
        else:

            q_values = self.predictor(old_s[None, ...])[0][0]
            q_values[mask == 0] = np.nan
            act = np.nanargmax(q_values)
            assert act < self.num_actions
        reward, isOver, _ = self.player.step_manual(to_value(action_space[act]))

        # step for AI
        while not isOver and self.player.get_role_ID() != ROLE_ID_TO_TRAIN:
            _, reward, _ = self.player.step_auto()
            isOver = (reward != 0)
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
                self._current_ob = self.get_state()
                break
            self._current_game_score.reset()
        self._current_ob = self.get_state()
        self.mem.append(Experience(old_s, mask, act, reward, isOver))

    def debug(self, cnt=100000):
        with get_tqdm(total=cnt) as pbar:
            for i in range(cnt):
                self.mem.append(Experience(np.zeros([self.num_actions[0], self.num_actions[1], 256]), 0, 0))
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
        next_mask = np.asarray([e[1] for e in batch_exp], dtype='bool')
        action = np.asarray([e[2] for e in batch_exp], dtype='int32')
        reward = np.asarray([e[3] for e in batch_exp], dtype='float32')
        isOver = np.asarray([e[4] for e in batch_exp], dtype='bool')
        return [state, next_mask, action, reward, isOver]

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
