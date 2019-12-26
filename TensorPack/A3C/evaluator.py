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

import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

from env import Env
from utils import to_char
from card import Card, action_space, Category
import numpy as np
import tensorflow as tf
from utils import get_mask, get_minor_cards, train_fake_action_60, get_masks, test_fake_action
from utils import get_seq_length, pick_minor_targets, to_char, to_value, get_mask_alter, discard_onehot_from_s_60
from utils import inference_minor_cards, gputimeblock, give_cards_without_minor, pick_main_cards
from TensorPack.A3C.simulator import ROLE_IDS_TO_TRAIN


def play_one_episode(env, func):
    def take_action_from_prob(prob, mask):
        prob = prob[0]
        # to avoid numeric difficulty
        prob[mask == 0] = -1
        return np.argmax(prob)

    env.reset()
    # init_cards = np.arange(52)
    # init_cards = np.append(init_cards[::4], init_cards[1::4])
    # env.prepare_manual(init_cards)
    env.prepare()
    r = 0
    lstm_state = np.zeros([1024 * 2])
    while r == 0:
        last_cards_value = env.get_last_outcards()
        last_cards_char = to_char(last_cards_value)
        last_two_cards = env.get_last_two_cards()
        last_two_cards_onehot = np.concatenate(
            [Card.val2onehot60(last_two_cards[0]), Card.val2onehot60(last_two_cards[1])])
        curr_cards_char = to_char(env.get_curr_handcards())
        is_active = True if last_cards_value.size == 0 else False

        s = env.get_state_prob()
        s = np.concatenate([Card.char2onehot60(curr_cards_char), s])
        # print(s.shape)

        role_id = env.get_role_ID()
        # print('%s current cards' % ('lord' if role_id == 2 else 'farmer'), curr_cards_char)

        if role_id in ROLE_IDS_TO_TRAIN:
            if is_active:
                # first get mask
                mask = get_mask(curr_cards_char, action_space, None)
                # not valid for active
                mask[0] = 0

                active_prob, _, lstm_state = func(np.array([role_id]), s.reshape(1, -1), np.zeros([1, 120]), lstm_state.reshape(1, -1))

                # make decision depending on output
                action_idx = take_action_from_prob(active_prob, mask)
            else:
                # print('last cards char', last_cards_char)
                mask = get_mask(curr_cards_char, action_space, last_cards_char)

                _, passive_prob, lstm_state = func(np.array([role_id]), s.reshape(1, -1), last_two_cards_onehot.reshape(1, -1), lstm_state.reshape(1, -1))

                action_idx = take_action_from_prob(passive_prob, mask)

            # since step auto needs full last card group info, we do not explicitly feed card type
            intention = to_value(action_space[action_idx])
            r, _, _ = env.step_manual(intention)
            # print('lord gives', to_char(intention))
            assert (intention is not None)
        else:
            intention, r, _ = env.step_auto()
            # print('farmer gives', to_char(intention))
    # if r > 0:
    #     print('farmer wins')
    # else:
    #     print('lord wins')
    return int(r > 0)


def eval_with_funcs(predictors, nr_eval, get_player_fn, verbose=False):
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
                        val = play_one_episode(player, self.func)
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
    def __init__(self, nr_eval, input_names, output_names, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        # self.lord_win_rate = tf.get_variable('lord_win_rate', shape=[], initializer=tf.constant_initializer(0.),
        #                trainable=False)
        nr_proc = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * nr_proc

    def _before_train(self):
        t = time.time()
        farmer_win_rate = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, verbose=False)
        t = time.time() - t
        logger.info("farmer win rate: {}".format(farmer_win_rate))
        logger.info("lord win rate: {}".format(1 - farmer_win_rate))
        # self.lord_win_rate.load(1 - farmer_win_rate)
        # if t > 10 * 60:  # eval takes too long
        #     self.eval_episode = int(self.eval_episode * 0.94)

    def _trigger_epoch(self):
        t = time.time()
        farmer_win_rate = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, verbose=False)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('farmer_win_rate', farmer_win_rate)
        self.trainer.monitors.put_scalar('lord_win_rate', 1 - farmer_win_rate)


if __name__ == '__main__':
    env = Env()
    stat = StatCounter()
    init_cards = np.arange(36)
    # init_cards = np.append(init_cards[::4], init_cards[1::4])
    for _ in range(100):
        env.reset()
        env.prepare_manual(init_cards)
        r = 0
        while r == 0:
            _, r, _ = env.step_auto()
        stat.feed(int(r < 0))
    print('lord win rate: {}'.format(stat.average))
