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

from env import Env, get_combinations_nosplit, get_combinations_recursive
from card import Card, action_space, action_space_onehot60, Category, CardGroup, augment_action_space_onehot60, augment_action_space, clamp_action_idx
import numpy as np
import tensorflow as tf
from utils import get_mask, get_minor_cards, train_fake_action_60, get_masks, test_fake_action
from utils import get_seq_length, pick_minor_targets, to_char, to_value, get_mask_alter, get_mask_onehot60
from utils import inference_minor_cards, gputimeblock, give_cards_without_minor, pick_main_cards
from TensorPack.Vanilla_Q.expreplay import ROLE_ID_TO_TRAIN


encoding = None


def get_state(env):
    def cards_char2embedding(cards_char):
        test = (action_space_onehot60 == Card.char2onehot60(cards_char))
        test = np.all(test, axis=1)
        target = np.where(test)[0]
        return encoding[target[0]]

    s = env.get_state_prob()
    s = np.concatenate([Card.val2onehot60(env.get_curr_handcards()), s])
    last_two_cards_char = env.get_last_two_cards()
    last_two_cards_char = [to_char(c) for c in last_two_cards_char]
    return np.concatenate(
        [s, cards_char2embedding(last_two_cards_char[0]), cards_char2embedding(last_two_cards_char[1])])


def play_one_episode(env, func):

    env.reset()
    env.prepare()
    r = 0
    while r == 0:
        role_id = env.get_role_ID()
        if role_id == ROLE_ID_TO_TRAIN:
            s = get_state(env)
            mask = get_mask(to_char(env.get_curr_handcards()), action_space,
                            to_char(env.get_last_outcards()))
            q_values = func(s[None, ...])[0][0]
            q_values[mask == 0] = np.nan
            act = np.nanargmax(q_values)
            intention = to_value(action_space[act])
            r, _, _ = env.step_manual(intention)
        else:
            intention, r, _ = env.step_auto()
    return int(r > 0)


def eval_with_funcs(predictors, nr_eval, get_player_fn, verbose=False):
    """
    Args:
        predictors ([PredictorBase])
    """

    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self.func = func
            self.q = queue

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
        global encoding
        if encoding is None:
            encoding = np.load('../AutoEncoder/encoding.npy')
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        # self.lord_win_rate = tf.get_variable('lord_win_rate', shape=[], initializer=tf.constant_initializer(0.),
        #                trainable=False)
        nr_proc = min(multiprocessing.cpu_count() // 2, 1)
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
    # encoding = np.load('encoding.npy')
    # print(encoding.shape)
    # env = Env()
    # stat = StatCounter()
    # init_cards = np.arange(21)
    # # init_cards = np.append(init_cards[::4], init_cards[1::4])
    # for _ in range(10):
    #     fw = play_one_episode(env, lambda b: np.random.rand(1, 1, 100) if b[1][0] else np.random.rand(1, 1, 21), [100, 21])
    #     stat.feed(int(fw))
    # print('lord win rate: {}'.format(1. - stat.average))
    env = Env()
    stat = StatCounter()
    for i in range(100):
        env.reset()
        print('begin')
        env.prepare()
        r = 0
        while r == 0:
            role = env.get_role_ID()
            intention, r, _ = env.step_auto()
            # print('lord gives' if role == 2 else 'farmer gives', to_char(intention))
        stat.feed(int(r < 0))
    print(stat.average)
