import random
import time
import multiprocessing
from tqdm import tqdm
from six.moves import queue

from tensorpack.utils.concurrency import StoppableThread, ShareSessionThread
from tensorpack.callbacks import Callback
from tensorpack.utils import logger
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs

import sys
import os

if os.name == 'nt':
    sys.path.insert(0, '../../build/Release')
else:
    sys.path.insert(0, '../../build.linux')
from env import Env, get_combinations_nosplit, get_combinations_recursive
from logger import Logger
from utils import to_char
from card import Card, action_space, Category
import numpy as np
import tensorflow as tf
from utils import get_mask, get_minor_cards, train_fake_action_60, get_masks, test_fake_action
from utils import get_seq_length, pick_minor_targets, to_char, to_value, get_mask_alter, get_mask_onehot60
from utils import inference_minor_cards, gputimeblock, give_cards_without_minor, pick_main_cards


encoding = np.load('encoding.npy')


def play_one_episode(env, func, num_actions):
    def pad_state(state):
        # since out net uses max operation, we just dup the last row and keep the result same
        newstates = []
        for s in state:
            assert s.shape[0] <= num_actions[1]
            s = np.concatenate([s, np.repeat(s[-1:, :], num_actions[1] - s.shape[0], axis=0)], axis=0)
            newstates.append(s)
        newstates = np.stack(newstates, axis=0)
        if len(state) < num_actions[0]:
            state = np.concatenate([newstates, np.repeat(newstates[-1:, :, :], num_actions[0] - newstates.shape[0], axis=0)], axis=0)
        else:
            state = newstates
        return state

    def pad_action_space(available_actions):
        # print(available_actions)
        for i in range(len(available_actions)):
            available_actions[i] += [available_actions[i][-1]] * (num_actions[1] - len(available_actions[i]))
        if len(available_actions) < num_actions[0]:
            available_actions.extend([available_actions[-1]] * (num_actions[0] - len(available_actions)))

    def get_combinations(curr_cards_char, last_cards_value):
        if len(curr_cards_char) > 10:
            mask = get_mask_onehot60(curr_cards_char, action_space,
                                     None if last_cards_value.size == 0 else to_char(last_cards_value)).astype(
                np.uint8)
            combs = get_combinations_nosplit(mask, Card.char2onehot60(curr_cards_char).astype(np.uint8))
        else:
            mask = get_mask_onehot60(curr_cards_char, action_space,
                                     None if last_cards_value.size == 0 else to_char(last_cards_value))\
                .reshape(len(action_space), 15, 4).sum(-1).astype(np.uint8)
            valid = mask.sum(-1) > 0
            cards_target = Card.char2onehot60(curr_cards_char).reshape(-1, 4).sum(-1).astype(np.uint8)
            combs = get_combinations_recursive(mask, cards_target, valid)
        return combs

    def subsample_combs(combs, num_sample):
        random.shuffle(combs)
        return combs[:num_sample]

    def get_state_and_action_space(is_comb, cand_state=None, cand_actions=None, action=None):
        if is_comb:
            combs = get_combinations(curr_cards_char, last_cards_value)
            if len(combs) > num_actions[0]:
                combs = subsample_combs(combs, num_actions[0])
            # TODO: utilize temporal relations to speedup
            available_actions = [([[]] if last_cards_value.size > 0 else []) + [action_space[idx] for idx in comb] for
                                 comb in combs]
            if len(combs) == 0:
                available_actions = [[[]]]
            pad_action_space(available_actions)
            if len(combs) < num_actions[0]:
                available_actions.extend([available_actions[-1]] * (num_actions[0] - len(combs)))
            state = [np.stack(
                ([encoding[0]] if last_cards_value.size > 0 else []) + [encoding[idx] for idx in comb])
                for comb in combs]
            if len(state) == 0:
                assert len(combs) == 0
                state = np.array([[encoding[0]]])
            state = pad_state(state)
            assert state.shape[0] == num_actions[0] and state.shape[1] == num_actions[1]
        else:
            assert action is not None
            available_actions = cand_actions[action]
            state = cand_state[action:action + 1, :, :]
            # print(cand_state.shape)
            # print(action)
            state = np.repeat(state, num_actions[0], axis=0)
            assert state.shape[0] == num_actions[0] and state.shape[1] == num_actions[1]
        return state, available_actions

    env.reset()
    init_cards = np.arange(21)
    # init_cards = np.append(init_cards[::4], init_cards[1::4])
    env.prepare_manual(init_cards)
    # env.prepare()
    r = 0
    while r == 0:
        role_id = env.get_role_ID()

        if role_id == 2:
            last_cards_value = env.get_last_outcards()
            is_active = True if last_cards_value.size == 0 else False
            curr_cards_char = to_char(env.get_curr_handcards())
            # print('%s current cards' % ('lord' if role_id == 2 else 'farmer'), curr_cards_char)

            # first hierarchy
            state, available_actions = get_state_and_action_space(True)
            q_values = func([state[None, :, :, :], np.array([True])])[0][0]
            action = np.argmax(q_values)
            # clamp action to valid range
            action = min(action, num_actions[0] - 1)

            # second hierarchy
            state, available_actions = get_state_and_action_space(False, state, available_actions, action)
            q_values = func([state[None, :, :, :], np.array([False])])[0][0]
            action = np.argmax(q_values)
            # clamp action to valid range
            action = min(action, num_actions[1] - 1)

            # intention
            intention = to_value(available_actions[action])
            r, _, _ = env.step_manual(intention)
            # print('lord gives', to_char(intention))
            assert (intention is not None)
        else:
            intention, r, _ = env.step_auto()
    # if r > 0:
    #     print('farmer wins')
    # else:
    #     print('lord wins')
    return int(r > 0)


def eval_with_funcs(predictors, nr_eval, get_player_fn, num_actions, verbose=False):
    """
    Args:
        predictors ([PredictorBase])
    """

    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn()
                while not self.stopped():
                    try:
                        val = play_one_episode(player, self.func, num_actions)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, val)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()

    def fetch():
        val = q.get()
        stat.feed(val)
        if verbose:
            if val > 0:
                logger.info("farmer wins")
            else:
                logger.info("lord wins")

    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        fetch()
    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        fetch()
    farmer_win_rate = stat.average
    return farmer_win_rate


class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names, num_actions, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn
        self.num_actions = num_actions

    def _setup_graph(self):
        # self.lord_win_rate = tf.get_variable('lord_win_rate', shape=[], initializer=tf.constant_initializer(0.),
        #                trainable=False)
        nr_proc = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * nr_proc

    def _before_train(self):
        t = time.time()
        farmer_win_rate = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, self.num_actions, verbose=False)
        t = time.time() - t
        logger.info("farmer win rate: {}".format(farmer_win_rate))
        logger.info("lord win rate: {}".format(1 - farmer_win_rate))
        # self.lord_win_rate.load(1 - farmer_win_rate)
        # if t > 10 * 60:  # eval takes too long
        #     self.eval_episode = int(self.eval_episode * 0.94)

    def _trigger_epoch(self):
        t = time.time()
        farmer_win_rate = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, self.num_actions, verbose=False)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('farmer win rate', farmer_win_rate)
        self.trainer.monitors.put_scalar('lord win rate', 1 - farmer_win_rate)


if __name__ == '__main__':
    env = Env()
    stat = StatCounter()
    init_cards = np.arange(52)
    init_cards = np.append(init_cards[::4], init_cards[1::4])
    for _ in range(1000):
        env.reset()
        env.prepare_manual(init_cards)
        r = 0
        while r == 0:
            _, r, _ = env.step_auto()
        stat.feed(int(r < 0))
    print('lord win rate: {}'.format(stat.average))
