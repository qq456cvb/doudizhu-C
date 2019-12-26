import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))
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

from env import Env, get_combinations_nosplit, get_combinations_recursive
from card import Card, action_space, action_space_onehot60, Category, CardGroup, augment_action_space_onehot60, augment_action_space, clamp_action_idx
from utils import to_char, to_value
import numpy as np
from TensorPack.MA_Hierarchical_Q.predictor import Predictor


encoding = np.load(os.path.join(ROOT_PATH, 'TensorPack/AutoEncoder/encoding.npy'))


def play_one_episode(env, func, role_id):

    env.reset()
    env.prepare()
    r = 0
    while r == 0:
        if env.get_role_ID() == role_id:
            handcards = to_char(env.get_curr_handcards())
            last_two_cards = env.get_last_two_cards()
            last_two_cards = [to_char(cards) for cards in last_two_cards]
            prob_state = env.get_state_prob()
            # print(agent, handcards)

            action = func.predict(handcards, last_two_cards, prob_state)
            # print(agent, ' gives ', action)
            intention = to_value(action)
            r, _, _ = env.step_manual(intention)
            # print('lord gives', to_char(intention), file=f)
            assert (intention is not None)
        else:
            intention, r, _ = env.step_auto()

    return int(r > 0)


def eval_with_funcs(predictors, role_id, nr_eval, get_player_fn):
    """
    Args:
        predictors ([PredictorBase])
    """

    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, role_id, queue):
            super(Worker, self).__init__()
            self.func = func
            self.role_id = role_id
            self.q = queue

        def run(self):
            with self.default_sess():
                player = get_player_fn()
                while not self.stopped():
                    try:
                        val = play_one_episode(player, self.func, self.role_id)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, val)

    q = queue.Queue()
    threads = [Worker(f, role_id, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()

    def fetch():
        val = q.get()
        stat.feed(val)

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


class BLEvaluator(Callback):
    def __init__(self, nr_eval, agent_name, role_id, get_player_fn):
        self.eval_episode = nr_eval
        self.agent_name = agent_name
        self.role_id = role_id
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        # self.lord_win_rate = tf.get_variable('lord_win_rate', shape=[], initializer=tf.constant_initializer(0.),
        #                trainable=False)
        nr_proc = min(multiprocessing.cpu_count() // 2, 10)
        self.predictor = Predictor(
            self.trainer.get_predictor([self.agent_name + '/state:0', self.agent_name + '_comb_mask:0', self.agent_name + '/fine_mask:0'], [self.agent_name + '/Qvalue:0']))
        self.pred_funcs = [self.predictor] * nr_proc

    def _before_train(self):
        t = time.time()
        farmer_win_rate = eval_with_funcs(
            self.pred_funcs, self.role_id, self.eval_episode, self.get_player_fn)
        t = time.time() - t
        logger.info(("ROLE_ID_[%d]_farmer_win_rate: {}" % self.role_id).format(farmer_win_rate))
        logger.info(("ROLE_ID_[%d]_lord_win_rate: {}" % self.role_id).format(1 - farmer_win_rate))
        # self.lord_win_rate.load(1 - farmer_win_rate)
        # if t > 10 * 60:  # eval takes too long
        #     self.eval_episode = int(self.eval_episode * 0.94)

    def _trigger_epoch(self):
        t = time.time()
        farmer_win_rate = eval_with_funcs(
            self.pred_funcs, self.role_id, self.eval_episode, self.get_player_fn)
        t = time.time() - t
        # if t > 10 * 60:  # eval takes too long
        #     self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('ROLE_ID_[%d]_farmer_win_rate' % self.role_id, farmer_win_rate)
        self.trainer.monitors.put_scalar('ROLE_ID_[%d]_lord_win_rate' % self.role_id, 1 - farmer_win_rate)


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
