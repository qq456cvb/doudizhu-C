#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: expreplay.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Adapted by: Neil You for Fight the Lord

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
from card import action_space, Card
from utils import to_value, to_char, get_mask_onehot60
import sys
import os
if os.name == 'nt':
    sys.path.insert(0, '../../build/Release')
else:
    sys.path.insert(0, '../../build.linux')
from env import Env as CEnv
from env import get_combinations_nosplit

__all__ = ['ExpReplay']

Experience = namedtuple('Experience',
                        ['joint_state', 'action', 'reward', 'isOver', 'comb_mask'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape):
        self.max_size = int(max_size)
        self.state_shape = state_shape

        self.state = np.zeros((self.max_size,) + state_shape, dtype='float32')
        self.action = np.zeros((self.max_size,), dtype='int32')
        self.reward = np.zeros((self.max_size,), dtype='float32')
        self.isOver = np.zeros((self.max_size,), dtype='bool')
        self.comb_mask = np.zeros((self.max_size,), dtype='bool')

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
        if idx + 2 <= self._curr_size:
            state = self.state[idx:idx+2]
            reward = self.reward[idx]
            action = self.action[idx]
            isOver = self.isOver[idx]
            comb_mask = self.comb_mask[idx]
        else:
            end = idx + 2 - self._curr_size
            state = self._slice(self.state, idx, end)
            reward = self._slice(self.reward, idx, end)
            action = self._slice(self.action, idx, end)
            isOver = self._slice(self.isOver, idx, end)
            comb_mask = self._slice(self.comb_mask, idx, end)
        return state, reward, action, isOver, comb_mask

    def _slice(self, arr, start, end):
        s1 = arr[start:]
        s2 = arr[:end]
        return np.concatenate((s1, s2), axis=0)

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        self.state[pos] = exp.joint_state
        self.reward[pos] = exp.reward
        self.action[pos] = exp.action
        self.isOver[pos] = exp.isOver
        self.comb_mask[pos] = exp.comb_mask


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
                 encoding_file='encoding.npy'):
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
        init_cards = np.arange(21)
        self.player.prepare_manual(init_cards)
        # self._current_ob = self.player.get_state_prob()
        self._comb_mask = True
        self._current_ob, self._action_space = self.get_state_and_action_spaces()
        self._player_scores = StatCounter()
        self._current_game_score = StatCounter()

    def get_state_and_action_spaces(self, action=None):
        last_cards_value = self.player.get_last_outcards()
        curr_cards_char = to_char(self.player.get_curr_handcards())
        if self._comb_mask:
            mask = get_mask_onehot60(curr_cards_char, action_space, None if last_cards_value.size == 0 else to_char(last_cards_value)).astype(np.uint8)
            combs = get_combinations_nosplit(mask, Card.char2onehot60(curr_cards_char).astype(np.uint8))
            if len(combs) == 0:
                # we have no larger cards
                assert last_cards_value.size > 0
            if len(combs) > self.num_actions[0]:
                combs = combs[:self.num_actions[0]]
            # TODO: utilize temporal relations to speedup
            available_actions = [([[]] if last_cards_value.size > 0 else []) + [action_space[idx] for idx in comb] for comb in combs]
            if len(combs) == 0:
                available_actions = [[[]]]
            self.pad_action_space(available_actions)
            state = [np.stack(([self.encoding[0]] if last_cards_value.size > 0 else []) + [self.encoding[idx] for idx in comb]) for comb in combs]
            if len(state) == 0:
                assert len(combs) == 0
                state = np.array([[self.encoding[0]]])
            state = self.pad_state(state)
            assert state.shape[0] == self.num_actions[0] and state.shape[1] == self.num_actions[1]
        else:
            assert action is not None
            available_actions = self._action_space[action]
            state = self._current_ob[action:action+1, :, :]
            state = np.repeat(state, self.num_actions[0], axis=0)
            assert state.shape[0] == self.num_actions[0] and state.shape[1] == self.num_actions[1]
        return state, available_actions

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
        if self.rng.rand() <= self.exploration:
            act = self.rng.choice(range(self.num_actions[0 if comb_mask else 1]))
        else:
            q_values = self.predictor([old_s[None, :, :, :], np.array([comb_mask])])[0][0]
            act = np.argmax(q_values)
            # clamp action to valid range
            act = min(act, self.num_actions[0 if comb_mask else 1] - 1)
        if comb_mask:
            reward = 0
            isOver = False
        else:
            reward, isOver, _ = self.player.step_manual(to_value(self._action_space[act]))

        # step for AI farmers
        while not isOver and self.player.get_role_ID() != 2:
            _, reward, _ = self.player.step_auto()
            isOver = (reward != 0)
        reward = -reward
        self._current_game_score.feed(reward)

        if isOver:
            self._player_scores.feed(self._current_game_score.sum)
            self.player.reset()
            init_cards = np.arange(21)
            self.player.prepare_manual(init_cards)
            self._comb_mask = True
            self._current_game_score.reset()
        else:
            self._comb_mask = not self._comb_mask
        self._current_ob, self._action_space = self.get_state_and_action_spaces(act if not self._comb_mask else None)
        self._current_game_score.feed(reward)
        self.mem.append(Experience(old_s, act, reward, isOver, comb_mask))

    def _debug_sample(self, sample):
        import cv2

        def view_state(comb_state):
            state = comb_state[:, :, :-1]
            next_state = comb_state[:, :, 1:]
            r = np.concatenate([state[:, :, k] for k in range(self.history_len)], axis=1)
            r2 = np.concatenate([next_state[:, :, k] for k in range(self.history_len)], axis=1)
            r = np.concatenate([r, r2], axis=0)
            cv2.imshow("state", r)
            cv2.waitKey()
        print("Act: ", sample[2], " reward:", sample[1], " isOver: ", sample[3])
        if sample[1] or sample[3]:
            view_state(sample[0])

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
        reward = np.asarray([e[1] for e in batch_exp], dtype='float32')
        action = np.asarray([e[2] for e in batch_exp], dtype='int32')
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        comb_mask = np.asarray([e[4] for e in batch_exp], dtype='bool')
        return [state, action, reward, isOver, comb_mask]

    def _setup_graph(self):
        self.predictor = self.trainer.get_predictor(*self.predictor_io_names)

    def _before_train(self):
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


def h():
    def f():
        print(a)
        print(b)
    a = 1
    b = 2
    f()
    a += 1
    b += 1
    f()


if __name__ == '__main__':
    def predictor(x):
        return [np.ones([1, 21])]
    player = CEnv()
    E = expreplay = ExpReplay(
        predictor_io_names=(['joint_state', 'comb_mask'], ['Qvalue']),
        player=player,
        state_shape=180,
        batch_size=4,
        memory_size=1e4,
        init_memory_size=1e3,
        init_exploration=0.5,
        update_frequency=4
    )
    E.predictor = predictor
    E._init_memory()

    for k in E.get_data():
        import IPython as IP
        IP.embed(config=IP.terminal.ipapp.load_default_config())
        pass
        # import IPython;
        # IPython.embed(config=IPython.terminal.ipapp.load_default_config())
        # break