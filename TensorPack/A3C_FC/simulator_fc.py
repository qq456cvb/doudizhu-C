#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: simulator.py
# Adapted by: Neil You on Fight the Lord
import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

import multiprocessing as mp
import time
import os
import threading
from abc import abstractmethod, ABCMeta
from collections import defaultdict

import six
from six.moves import queue
import zmq
import numpy as np

from tensorpack.utils import logger
from tensorpack.utils.serialize import loads, dumps
from tensorpack.utils.concurrency import LoopThread, ensure_proc_terminate
from utils import give_cards_without_minor, get_seq_length, get_mask_alter, to_char, get_masks, get_mask, discard_onehot_from_s_60
from card import Card, action_space
from card import Category


__all__ = ['SimulatorProcess', 'SimulatorMaster',
           'SimulatorProcessStateExchange',
           'TransitionExperience']


class MODE:
    PASSIVE_DECISION = 0
    PASSIVE_BOMB = 1
    PASSIVE_RESPONSE = 2
    ACTIVE_DECISION = 3
    ACTIVE_RESPONSE = 4
    ACTIVE_SEQ = 5
    MINOR_RESPONSE = 6


class ACT_TYPE:
    PASSIVE = 0
    ACTIVE = 1


class SubState:
    def __init__(self, act, all_state, curr_handcards_char, last_cards_value, last_category):
        self.act = act
        self.all_state = all_state
        self.finished = False
        self.mode = MODE.PASSIVE_DECISION if self.act == ACT_TYPE.PASSIVE else MODE.ACTIVE_DECISION
        self.intention = np.array([])
        self.last_cards_value = last_cards_value
        self.minor_type = 0
        self.category = last_category
        self.minor_length = 0
        self.curr_handcards_char = curr_handcards_char
        self.handcards_char = self.curr_handcards_char.copy()
        self.active_decision = 0
        self.active_response = 0
        self.card_type = -1

    @property
    def state(self):
        return get_mask(self.handcards_char, action_space, None if self.act == ACT_TYPE.ACTIVE else to_char(self.last_cards_value)).astype(np.float32)

    def get_mask(self):
        if self.act == ACT_TYPE.PASSIVE:
            decision_mask, response_mask, bomb_mask, _ = get_mask_alter(self.curr_handcards_char, to_char(self.last_cards_value), self.category)
            if self.mode == MODE.PASSIVE_DECISION:
                return decision_mask
            elif self.mode == MODE.PASSIVE_RESPONSE:
                return response_mask
            elif self.mode == MODE.PASSIVE_BOMB:
                return bomb_mask
            elif self.mode == MODE.MINOR_RESPONSE:
                input_single, input_pair, _, _ = get_masks(self.curr_handcards_char, None)
                if self.minor_type == 1:
                    mask = np.append(input_pair, [0, 0])
                else:
                    mask = input_single
                for v in set(self.intention):
                    mask[v - 3] = 0
                return mask
        elif self.act == ACT_TYPE.ACTIVE:
            decision_mask, response_mask, _, length_mask = get_mask_alter(self.curr_handcards_char, [], self.category)
            if self.mode == MODE.ACTIVE_DECISION:
                return decision_mask
            elif self.mode == MODE.ACTIVE_RESPONSE:
                return response_mask[self.active_decision]
            elif self.mode == MODE.ACTIVE_SEQ:
                return length_mask[self.active_decision][self.active_response]
            elif self.mode == MODE.MINOR_RESPONSE:
                input_single, input_pair, _, _ = get_masks(self.curr_handcards_char, None)
                if self.minor_type == 1:
                    mask = np.append(input_pair, [0, 0])
                else:
                    mask = input_single
                for v in set(self.intention):
                    mask[v - 3] = 0
                return mask

    def step(self, action):
        if self.act == ACT_TYPE.PASSIVE:
            if self.mode == MODE.PASSIVE_DECISION:
                if action == 0 or action == 2:
                    self.finished = True
                    if action == 2:
                        self.intention = np.array([16, 17])
                        self.card_type = Category.BIGBANG.value
                    else:
                        self.card_type = Category.EMPTY.value
                    return
                elif action == 1:
                    self.mode = MODE.PASSIVE_BOMB
                    return
                elif action == 3:
                    self.mode = MODE.PASSIVE_RESPONSE
                    return
                else:
                    raise Exception('unexpected action')
            elif self.mode == MODE.PASSIVE_BOMB:
                # convert to value input
                self.intention = np.array([action + 3] * 4)
                self.finished = True
                self.card_type = Category.QUADRIC.value
                return
            elif self.mode == MODE.PASSIVE_RESPONSE:
                self.intention = give_cards_without_minor(action, self.last_cards_value, self.category, None)
                if self.category == Category.THREE_ONE.value or \
                        self.category == Category.THREE_TWO.value or \
                        self.category == Category.THREE_ONE_LINE.value or \
                        self.category == Category.THREE_TWO_LINE.value or \
                        self.category == Category.FOUR_TWO.value:
                    if self.category == Category.THREE_TWO.value or self.category == Category.THREE_TWO_LINE.value:
                        self.minor_type = 1
                    self.mode = MODE.MINOR_RESPONSE
                    # modify the state for minor cards
                    # discard_onehot_from_s_60(self.prob_state, Card.val2onehot60(self.intention))
                    intention_char = to_char(self.intention)
                    for c in intention_char:
                        self.handcards_char.remove(c)
                    self.minor_length = get_seq_length(self.category, self.last_cards_value)
                    if self.minor_length is None:
                        self.minor_length = 2 if self.category == Category.FOUR_TWO.value else 1
                    self.card_type = self.category
                    return
                else:
                    self.finished = True
                    self.card_type = self.category
                    return
            elif self.mode == MODE.MINOR_RESPONSE:
                minor_value_cards = [action + 3] * (1 if self.minor_type == 0 else 2)
                # modify the state for minor cards
                minor_char = to_char(minor_value_cards)
                for c in minor_char:
                    self.handcards_char.remove(c)
                # discard_onehot_from_s_60(self.prob_state, Card.val2onehot60(minor_value_cards))
                self.intention = np.append(self.intention, minor_value_cards)
                assert self.minor_length > 0
                self.minor_length -= 1
                if self.minor_length == 0:
                    self.finished = True
                    return
                else:
                    return
        elif self.act == ACT_TYPE.ACTIVE:
            if self.mode == MODE.ACTIVE_DECISION:
                self.category = action + 1
                self.active_decision = action
                self.mode = MODE.ACTIVE_RESPONSE
                self.card_type = self.category
                return
            elif self.mode == MODE.ACTIVE_RESPONSE:
                if self.category == Category.SINGLE_LINE.value or \
                        self.category == Category.DOUBLE_LINE.value or \
                        self.category == Category.TRIPLE_LINE.value or \
                        self.category == Category.THREE_ONE_LINE.value or \
                        self.category == Category.THREE_TWO_LINE.value:
                    self.active_response = action
                    self.mode = MODE.ACTIVE_SEQ
                    return
                elif self.category == Category.THREE_ONE.value or \
                        self.category == Category.THREE_TWO.value or \
                        self.category == Category.FOUR_TWO.value:
                    if self.category == Category.THREE_TWO.value or self.category == Category.THREE_TWO_LINE.value:
                        self.minor_type = 1
                    self.mode = MODE.MINOR_RESPONSE
                    self.intention = give_cards_without_minor(action, np.array([]), self.category, None)
                    # modify the state for minor cards
                    intention_char = to_char(self.intention)
                    for c in intention_char:
                        self.handcards_char.remove(c)
                    # discard_onehot_from_s_60(self.prob_state, Card.val2onehot60(self.intention))
                    self.minor_length = 2 if self.category == Category.FOUR_TWO.value else 1
                    return
                else:
                    self.intention = give_cards_without_minor(action, np.array([]), self.category, None)
                    self.finished = True
                    return
            elif self.mode == MODE.ACTIVE_SEQ:
                self.minor_length = action + 1
                self.intention = give_cards_without_minor(self.active_response, np.array([]), self.category, action + 1)
                if self.category == Category.THREE_ONE_LINE.value or \
                        self.category == Category.THREE_TWO_LINE.value:
                    if self.category == Category.THREE_TWO.value or self.category == Category.THREE_TWO_LINE.value:
                        self.minor_type = 1
                    self.mode = MODE.MINOR_RESPONSE
                    # modify the state for minor cards
                    intention_char = to_char(self.intention)
                    for c in intention_char:
                        self.handcards_char.remove(c)
                    # discard_onehot_from_s_60(self.prob_state, Card.val2onehot60(self.intention))
                else:
                    self.finished = True
                return
            elif self.mode == MODE.MINOR_RESPONSE:
                minor_value_cards = [action + 3] * (1 if self.minor_type == 0 else 2)
                # modify the state for minor cards
                minor_char = to_char(minor_value_cards)
                for c in minor_char:
                    self.handcards_char.remove(c)
                # discard_onehot_from_s_60(self.prob_state, Card.val2onehot60(minor_value_cards))
                self.intention = np.append(self.intention, minor_value_cards)
                assert self.minor_length > 0
                self.minor_length -= 1
                if self.minor_length == 0:
                    self.finished = True
                    return
                else:
                    return


class TransitionExperience(object):
    """ A transition of state, or experience"""

    def __init__(self, prob_state, all_state, action, reward, **kwargs):
        """ kwargs: whatever other attribute you want to save"""
        self.prob_state = prob_state
        self.all_state = all_state
        self.action = action
        self.reward = reward
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)


@six.add_metaclass(ABCMeta)
class SimulatorProcessBase(mp.Process):
    def __init__(self, idx):
        super(SimulatorProcessBase, self).__init__()
        self.idx = int(idx)
        self.name = u'simulator-{}'.format(self.idx)
        self.identity = self.name.encode('utf-8')

    @abstractmethod
    def _build_player(self):
        pass


class SimulatorProcessStateExchange(SimulatorProcessBase):
    """
    A process that simulates a player and communicates to master to
    send states and receive the next action
    """

    def __init__(self, idx, pipe_c2s, pipe_s2c):
        """
        Args:
            idx: idx of this process
            pipe_c2s, pipe_s2c (str): name of the pipe
        """
        super(SimulatorProcessStateExchange, self).__init__(idx)
        self.c2s = pipe_c2s
        self.s2c = pipe_s2c

    def run(self):
        player = self._build_player()
        context = zmq.Context()
        c2s_socket = context.socket(zmq.PUSH)
        c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        c2s_socket.set_hwm(10)
        c2s_socket.connect(self.c2s)

        s2c_socket = context.socket(zmq.DEALER)
        s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
        s2c_socket.connect(self.s2c)

        player.reset()
        init_cards = np.arange(21)
        # init_cards = np.append(init_cards[::4], init_cards[1::4])
        player.prepare_manual(init_cards)
        r, is_over = 0, False
        while True:
            all_state, role_id, curr_handcards_value, last_cards_value, last_category = \
                player.get_state_all_cards(), player.get_role_ID(), player.get_curr_handcards(), player.get_last_outcards(), player.get_last_outcategory_idx()
            # after taking the last action, get to this state and get this reward/isOver.
            # If isOver, get to the next-episode state immediately.
            # This tuple is not the same as the one put into the memory buffer
            is_active = (last_cards_value.size == 0)
            all_state = np.stack([get_mask(Card.onehot2char(all_state[i*60:(i+1)*60]), action_space, None if is_active else to_char(last_cards_value)).astype(np.float32) for i in range(3)]).reshape(-1)
            last_state = get_mask(to_char(last_cards_value), action_space, None).astype(np.float32)

            if role_id == 2:
                st = SubState(ACT_TYPE.PASSIVE if last_cards_value.size > 0 else ACT_TYPE.ACTIVE, all_state,
                              to_char(curr_handcards_value), last_cards_value, last_category)
                if last_cards_value.size > 0:
                    assert last_category > 0
                first_st = True
                while not st.finished:
                    c2s_socket.send(dumps(
                        (self.identity, role_id, st.state, st.all_state, last_state, first_st, st.get_mask(), st.minor_type, st.mode, r, is_over)),
                        copy=False)
                    first_st = False
                    action = loads(s2c_socket.recv(copy=False).bytes)
                    # logger.info('received action {}'.format(action))
                    # print(action)
                    st.step(action)

                # print(st.intention)
                assert st.card_type != -1
                r, is_over, category_idx = player.step_manual(st.intention)
            else:
                _, r, _ = player.step_auto()
                is_over = (r != 0)
            if is_over:
                # print('{} over with reward {}'.format(self.identity, r))
                # logger.info('{} over with reward {}'.format(self.identity, r))
                # sys.stdout.flush()
                player.reset()
                player.prepare_manual(init_cards)



# compatibility
SimulatorProcess = SimulatorProcessStateExchange


class SimulatorMaster(threading.Thread):
    """ A base thread to communicate with all StateExchangeSimulatorProcess.
        It should produce action for each simulator, as well as
        defining callbacks when a transition or an episode is finished.
    """
    class ClientState(object):
        def __init__(self):
            self.memory = [[] for _ in range(3)]    # list of Experience
            self.ident = None

    def __init__(self, pipe_c2s, pipe_s2c):
        super(SimulatorMaster, self).__init__()
        assert os.name != 'nt', "Doesn't support windows!"
        self.daemon = True
        self.name = 'SimulatorMaster'

        self.context = zmq.Context()

        self.c2s_socket = self.context.socket(zmq.PULL)
        self.c2s_socket.bind(pipe_c2s)
        self.c2s_socket.set_hwm(20)
        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(pipe_s2c)
        self.s2c_socket.set_hwm(20)

        # queueing messages to client
        self.send_queue = queue.Queue(maxsize=1000)

        def f():
            msg = self.send_queue.get()
            self.s2c_socket.send_multipart(msg, copy=False)
        self.send_thread = LoopThread(f)
        self.send_thread.daemon = True
        self.send_thread.start()

        # make sure socket get closed at the end
        def clean_context(soks, context):
            for s in soks:
                s.close()
            context.term()
        import atexit
        atexit.register(clean_context, [self.c2s_socket, self.s2c_socket], self.context)

    def run(self):
        self.clients = defaultdict(self.ClientState)
        try:
            while True:
                msg = loads(self.c2s_socket.recv(copy=False).bytes)
                ident, role_id, prob_state, all_state, last_cards, first_st, mask, minor_type, mode, reward, isOver = msg
                client = self.clients[ident]
                if client.ident is None:
                    client.ident = ident
                # maybe check history and warn about dead client?
                self._process_msg(client, role_id, prob_state, all_state, last_cards, first_st, mask, minor_type, mode, reward, isOver)
        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")

    # @abstractmethod
    # def _process_msg(self, client, state, reward, isOver):
    #     pass

    def __del__(self):
        self.context.destroy(linger=0)


if __name__ == '__main__':

    class NaiveSimulator(SimulatorProcess):
        def _build_player(self):
            return CEnv()

    class NaiveActioner(SimulatorMaster):
        def _process_msg(self, client, role_id, prob_state, all_state, last_cards_onehot, first_st, mask, minor_type,
                         mode, reward, isOver):
            """
            Process a message sent from some client.
            """
            # in the first message, only state is valid,
            # reward&isOver should be discarde
            # print('received msg')
            if isOver and first_st:
                # should clear client's memory and put to queue
                assert reward != 0
                for i in range(3):
                    j = -1
                    while client.memory[i][j].reward == 0:
                        # notice that C++ returns the reward for farmer, transform to the reward in each agent's perspective
                        client.memory[i][j].reward = reward if i != 1 else -reward
                        if client.memory[i][j].first_st:
                            break
                        j -= 1
                self._parse_memory(0, client)
            # feed state and return action
            rand_a = np.random.rand(mask.shape[0])
            rand_a = (rand_a + 1e-6) * mask
            self.send_queue.put([client.ident, dumps(np.argmax(rand_a))])
            client.memory[role_id - 1].append(TransitionExperience(
                prob_state, all_state, np.argmax(rand_a), reward=0, first_st=first_st, mode=mode))

        def _parse_memory(self, init_r, client):
            # for each agent's memory
            for role_id in range(1, 4):
                mem = client.memory[role_id - 1]

                mem.reverse()
                R = float(init_r)
                mem_valid = [m for m in mem if m.first_st]
                dr = []
                for idx, k in enumerate(mem_valid):
                    R = k.reward + 0.99 * R
                    dr.append(R)
                dr.reverse()
                # print(dr)
                mem.reverse()
                i = -1
                j = 0
                while j < len(mem):
                    if mem[j].first_st:
                        i += 1
                    target = [0 for _ in range(7)]
                    k = mem[j]
                    target[k.mode] = k.action
                    # self.queue.put(
                    #     [role_id, k.prob_state, k.all_state, k.last_cards_onehot, *target, k.minor_type, k.mode, k.prob,
                    #      dr[i]])
                    j += 1

                client.memory[role_id - 1] = []

        def _on_episode_over(self, client):
            # print("Over: ", client.memory)
            client.memory = []
            client.state = 0

    name = 'ipc://c2s'
    name2 = 'ipc://s2c'
    procs = [NaiveSimulator(k, name, name2) for k in range(20)]
    [k.start() for k in procs]

    th = NaiveActioner(name, name2)
    ensure_proc_terminate(procs)
    th.start()

    time.sleep(100)
