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
from utils import to_value
import sys
import os
if os.name == 'nt':
    sys.path.insert(0, '../../build/Release')
else:
    sys.path.insert(0, '../../build.linux')
from env import Env as CEnv

__all__ = ['ExpReplay']

Experience = namedtuple('Experience',
                        ['state', 'atten_state', 'action', 'reward', 'isOver', 'isActive'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape, atten_state_shape):
        self.max_size = int(max_size)
        self.state_shape = (state_shape,)

        self.state = np.zeros((self.max_size,) + (state_shape,), dtype='float32')
        self.atten_state = np.zeros((self.max_size,) + (atten_state_shape,), dtype='float32')
        self.action = np.zeros((self.max_size,), dtype='int32')
        self.reward = np.zeros((self.max_size,), dtype='float32')
        self.isOver = np.zeros((self.max_size,), dtype='bool')
        self.isActive = np.zeros((self.max_size,), dtype='bool')

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
            where s is of shape STATE_SIZE + (hist_len+1,)"""
        idx = (self._curr_pos + idx) % self._curr_size
        state = self.state[idx:idx+2]
        atten_state = self.atten_state[idx:idx+2]
        reward = self.reward[idx]
        action = self.action[idx]
        isOver = self.isOver[idx]
        isActive = self.isActive[idx:idx+2]
        return state, atten_state, reward, action, isOver, isActive

    # the next_state is a different episode if current_state.isOver==True
    def _pad_sample(self, state, reward, action, isOver):
        for k in range(self.history_len - 2, -1, -1):
            if isOver[k]:
                state = copy.deepcopy(state)
                state[:k + 1].fill(0)
                break
        state = state.transpose(1, 2, 0)
        return (state, reward[-2], action[-2], isOver[-2])

    def _slice(self, arr, start, end):
        s1 = arr[start:]
        s2 = arr[:end]
        return np.concatenate((s1, s2), axis=0)

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        self.state[pos] = exp.state
        self.atten_state[pos] = exp.atten_state
        self.reward[pos] = exp.reward
        self.action[pos] = exp.action
        self.isOver[pos] = exp.isOver
        self.isActive[pos] = exp.isActive


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
                 atten_state_shape,
                 batch_size,
                 memory_size, init_memory_size,
                 init_exploration,
                 update_frequency):
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
        self.num_actions = len(action_space)
        logger.info("Number of Legal actions: {}".format(self.num_actions))

        self.rng = get_rng(self)
        self._init_memory_flag = threading.Event()  # tell if memory has been initialized

        # a queue to receive notifications to populate memory
        self._populate_job_queue = queue.Queue(maxsize=5)

        self.mem = ReplayMemory(memory_size, state_shape, atten_state_shape)
        self.player.reset()
        self.player.prepare()
        self._current_ob = self.player.get_state_prob()
        self._atten_ob = Card.val2onehot60(self.player.get_last_outcards())
        self._is_active = True
        self._player_scores = StatCounter()
        self._current_game_score = StatCounter()

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
        old_att = self._atten_ob
        is_active = self._is_active
        if self.rng.rand() <= self.exploration:
            act = self.rng.choice(range(self.num_actions))
        else:
            # assume batched network
            state = np.stack([old_s, np.zeros_like(old_s)], axis=0)
            atten_state = np.stack([old_att, np.zeros_like(old_att)], axis=0)
            q_values = self.predictor([state[None, :, :], atten_state[None, :, :], [is_active, is_active]])[0][0]  # this is the bottleneck
            act = np.argmax(q_values)
        reward, isOver, _ = self.player.step_manual(to_value(action_space[act]))
        while not isOver and self.player.get_role_ID() != 2:
            _, reward, _ = self.player.step_auto()
            isOver = (reward != 0)
        # get reward for lord
        reward = -reward

        if isOver:
            self.player.reset()
            self.player.prepare()
        self._current_ob = self.player.get_state_prob()
        last_cards_val = self.player.get_last_outcards()
        self._atten_ob = Card.val2onehot60(last_cards_val)
        self._is_active = (last_cards_val.size == 0)
        self._current_game_score.feed(reward)
        self.mem.append(Experience(old_s, old_att, act, reward, isOver, is_active))

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
        atten_state = np.asarray([e[1] for e in batch_exp], dtype='float32')
        reward = np.asarray([e[2] for e in batch_exp], dtype='float32')
        action = np.asarray([e[3] for e in batch_exp], dtype='int32')
        isOver = np.asarray([e[4] for e in batch_exp], dtype='bool')
        isActive = np.asarray([e[5] for e in batch_exp], dtype='bool')
        return [state, atten_state, action, reward, isOver, isActive]

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


if __name__ == '__main__':

    def predictor(x):
        return np.ones([1, len(action_space)])
    player = CEnv()
    E = expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
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