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
from env import Env
from logger import Logger
from utils import to_char
from card import Card, action_space, Category
import numpy as np
import tensorflow as tf
from utils import get_mask, get_minor_cards, train_fake_action_60, get_masks, test_fake_action
from utils import get_seq_length, pick_minor_targets, to_char, to_value, get_mask_alter, discard_onehot_from_s_60
from utils import inference_minor_cards, gputimeblock, give_cards_without_minor, pick_main_cards


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
    while r == 0:
        last_cards_value = env.get_last_outcards()
        last_cards_char = to_char(last_cards_value)
        last_out_cards = Card.val2onehot60(last_cards_value)
        last_category_idx = env.get_last_outcategory_idx()
        curr_cards_char = to_char(env.get_curr_handcards())
        is_active = True if last_cards_value.size == 0 else False

        s = env.get_state_prob()
        # print(s.shape)

        role_id = env.get_role_ID()
        # print('%s current cards' % ('lord' if role_id == 2 else 'farmer'), curr_cards_char)

        if role_id == 2:
            if is_active:
                # first get mask
                mask = get_mask(curr_cards_char, action_space, None)
                # not valid for active
                mask[0] = 0

                active_prob, _ = func(
                    [np.array([role_id]), s.reshape(1, -1), np.zeros([1, 60])]
                )

                # make decision depending on output
                action_idx = take_action_from_prob(active_prob, mask)
            else:
                # print('last cards char', last_cards_char)
                mask = get_mask(curr_cards_char, action_space, last_cards_char)

                _, passive_prob = func(
                    [np.array([role_id]), s.reshape(1, -1), last_out_cards.reshape(1, -1)])

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
<<<<<<< HEAD
        # self.lord_win_rate.load(1 - farmer_win_rate)
        self.trainer.monitors.put_scalar('farmer_win_rate', farmer_win_rate)
        self.trainer.monitors.put_scalar('lord_win_rate', 1 - farmer_win_rate)
=======
        self.trainer.monitors.put_scalar('farmer win rate', farmer_win_rate)
        self.trainer.monitors.put_scalar('lord win rate', 1 - farmer_win_rate)
>>>>>>> start from easy mode, modify resnet block to match pre-activation


if __name__ == '__main__':
    env = Env()
    stat = StatCounter()
<<<<<<< HEAD
    init_cards = np.arange(24)
    # init_cards = np.append(init_cards[::4], init_cards[1::4])
=======
    init_cards = np.arange(52)
    init_cards = np.append(init_cards[::4], init_cards[1::4])
>>>>>>> start from easy mode, modify resnet block to match pre-activation
    for _ in range(1000):
        env.reset()
        env.prepare_manual(init_cards)
        r = 0
        while r == 0:
            _, r, _ = env.step_auto()
        stat.feed(int(r < 0))
    print('lord win rate: {}'.format(stat.average))
