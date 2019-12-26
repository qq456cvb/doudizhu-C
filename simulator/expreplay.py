#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: expreplay.py
# Adapted by: Neil You for Fight the Lord
import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

import numpy as np
import copy
from collections import deque, namedtuple
import threading
from six.moves import queue, range
import queue

from tensorpack import *
from tensorpack.dataflow import DataFlow
from tensorpack.utils import logger
from tensorpack.utils.serialize import loads, dumps
from tensorpack.utils.utils import get_tqdm, get_rng
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.concurrency import LoopThread, ShareSessionThread
from tensorpack.callbacks.base import Callback
from card import action_space, action_space_onehot60, Card, CardGroup, augment_action_space_onehot60, clamp_action_idx, augment_action_space
from utils import to_char, to_value, get_mask_onehot60
import zmq
import time
import time
import random
from TensorPack.MA_Hierarchical_Q.predictor import Predictor
from env import get_combinations_nosplit, get_combinations_recursive


__all__ = ['ExpReplay']

Experience = namedtuple('Experience',
                        ['joint_state', 'action', 'reward', 'isOver', 'comb_mask', 'fine_mask'])


class ReplayMemory(object):
    def __init__(self, max_size, state_shape):
        self.max_size = int(max_size)
        self.state_shape = state_shape
        self.state = np.zeros((self.max_size, 2,) + state_shape, dtype='float32')
        self.action = np.zeros((self.max_size,), dtype='int32')
        self.reward = np.zeros((self.max_size,), dtype='float32')
        self.isOver = np.zeros((self.max_size,), dtype='bool')
        self.comb_mask = np.zeros((self.max_size,), dtype='bool')
        self.fine_mask = np.zeros((self.max_size, 2, max(state_shape[0], state_shape[1])), dtype='bool')

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
        state = self.state[idx]
        fine_mask = self.fine_mask[idx]
        # if idx + 2 <= self._curr_size:
        #     state = self.state[idx:idx+2]
        #     fine_mask = self.fine_mask[idx:idx+2]
        # else:
        #     end = idx + 2 - self._curr_size
        #     state = self._slice(self.state, idx, end)
        #     fine_mask = self._slice(self.fine_mask, idx, end)
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
                 # model,
                 agent_name,
                 state_shape,
                 num_actions,
                 batch_size,
                 memory_size, init_memory_size,
                 init_exploration,
                 update_frequency,
                 pipe_exp2sim, pipe_sim2exp):
        logger.info('starting expreplay {}'.format(agent_name))
        self.init_memory_size = int(init_memory_size)

        self.context = zmq.Context()
        # no reply for now
        # self.exp2sim_socket = self.context.socket(zmq.ROUTER)
        # self.exp2sim_socket.set_hwm(20)
        # self.exp2sim_socket.bind(pipe_exp2sim)

        self.sim2exp_socket = self.context.socket(zmq.PULL)
        self.sim2exp_socket.set_hwm(2)
        self.sim2exp_socket.bind(pipe_sim2exp)

        self.queue = queue.Queue(maxsize=1000)

        # self.model = model

        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.agent_name = agent_name

        self.exploration = init_exploration
        self.num_actions = num_actions
        logger.info("Number of Legal actions: {}, {}".format(*self.num_actions))

        self.rng = get_rng(self)
        self._init_memory_flag = threading.Event()  # tell if memory has been initialized

        # a queue to receive notifications to populate memory
        self._populate_job_queue = queue.Queue(maxsize=5)

        self.mem = ReplayMemory(memory_size, state_shape)
        # self._current_ob, self._action_space = self.get_state_and_action_spaces()
        self._player_scores = StatCounter()
        self._current_game_score = StatCounter()

    def get_recv_thread(self):
        def f():
            msg = self.sim2exp_socket.recv(copy=False).bytes
            msg = loads(msg)
            print('{}: received msg'.format(self.agent_name))
            try:
                self.queue.put_nowait(msg)
            except Exception:
                logger.info('put queue failed!')
            # send response or not?

        recv_thread = LoopThread(f, pausable=False)
        # recv_thread.daemon = True
        recv_thread.name = "recv thread"
        return recv_thread

    def get_simulator_thread(self):
        # spawn a separate thread to run policy
        def populate_job_func():
            self._populate_job_queue.get()
            i = 0
            # synchronous training
            while i < self.update_frequency:
                if self._populate_exp():
                    i += 1
                    time.sleep(0.1)

            # for _ in range(self.update_frequency):
            #     self._populate_exp()
        th = ShareSessionThread(LoopThread(populate_job_func, pausable=False))
        th.name = "SimulatorThread"
        return th

    def _init_memory(self):
        logger.info("{} populating replay memory with epsilon={} ...".format(self.agent_name, self.exploration))

        with get_tqdm(total=self.init_memory_size) as pbar:
            while len(self.mem) < self.init_memory_size:
                if self._populate_exp():
                    pbar.update()
        self._init_memory_flag.set()

    def _populate_exp(self):
        """ populate a transition by epsilon-greedy"""
        try:
            # do not wait for an update, this may cause some agents have old replay buffer trained more times before new buffer comes in
            state, action, reward, isOver, comb_mask, fine_mask = self.queue.get_nowait()
            self._current_game_score.feed(reward)
            # print(reward)

            if isOver:
                self._player_scores.feed(self._current_game_score.sum)
                self._current_game_score.reset()
            self.mem.append(Experience(np.stack(state), action, reward, isOver, comb_mask, np.stack(fine_mask)))
            return True
        except queue.Empty:
            return False

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
        self._recv_th = self.get_recv_thread()
        self._recv_th.start()
        # self.curr_predictor = self.trainer.get_predictor([self.agent_name + '/state:0', self.agent_name + '_comb_mask:0', self.agent_name + '/fine_mask:0'], [self.agent_name + '/Qvalue:0'])

    def _before_train(self):
        logger.info('{}-receive thread started'.format(self.agent_name))

        self._simulator_th = self.get_simulator_thread()
        self._simulator_th.start()

        self._init_memory()

    def _trigger(self):
        from simulator.tools import mean_score_logger
        v = self._player_scores
        try:
            mean, max = v.average, v.max
            logger.info('{} mean_score: {}'.format(self.agent_name, mean))
            mean_score_logger('{} mean_score: {}\n'.format(self.agent_name, mean))
            self.trainer.monitors.put_scalar('expreplay/mean_score', mean)
            self.trainer.monitors.put_scalar('expreplay/max_score', max)
        except Exception:
            logger.exception(self.agent_name + " Cannot log training scores.")
        v.reset()


if __name__ == '__main__':
    pass
    # def predictor(x):
    #     return [np.random.random([1, 100])]
    # player = CEnv()
    # E = ExpReplay(
    #     predictor_io_names=(['state', 'comb_mask'], ['Qvalue']),
    #     player=CEnv(),
    #     state_shape=(100, 21, 256),
    #     num_actions=[100, 21],
    #     batch_size=16,
    #     memory_size=1e4,
    #     init_memory_size=1e4,
    #     init_exploration=0.,
    #     update_frequency=4
    # )
    # E.predictor = predictor
    # E._init_memory()
    # for k in E.get_data():
    #     pass

    # for k in E.get_data():
    #     import IPython as IP
    #     IP.embed(config=IP.terminal.ipapp.load_default_config())
    #     pass
        # import IPython;
        # IPython.embed(config=IPython.terminal.ipapp.load_default_config())
        # break
