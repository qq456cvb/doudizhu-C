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
sys.path.insert(0, ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

from env import Env
from doudizhu.card import Card, action_space, action_index
from doudizhu.utils import to_char, get_mask
import numpy as np

LSTM_STATE_DIM = 1024 * 2


def policy_inputs(env):
    """Inputs of the policy network for the player to move, built the same way as in the A3C simulator.

    Returns (state, last_cards, mode): state is [own hand | card probabilities of the next and the other
    player] (60 * 3), last_cards the last two plays (60 * 2), and mode 0 when leading, 1 when responding.
    """
    state = np.concatenate([Card.val2onehot60(env.get_curr_handcards()), env.get_state_prob()])
    last_two_cards = env.get_last_two_cards()
    last_cards = np.concatenate([Card.val2onehot60(last_two_cards[0]), Card.val2onehot60(last_two_cards[1])])
    mode = 0 if env.get_last_outcards().size == 0 else 1
    return state, last_cards, mode


def play_one_episode(env, func):
    """Let the rule-based players play a game and check whether the policy picks the same move.

    Returns StatCounters of that accuracy when leading and when responding.
    """
    env.reset()
    env.prepare()
    r = 0
    stats = [StatCounter(), StatCounter()]
    lstm_state = np.zeros([1, LSTM_STATE_DIM])
    while r == 0:
        state, last_cards, mode = policy_inputs(env)
        handcards = to_char(env.get_curr_handcards())
        last_out_cards = to_char(env.get_last_outcards())

        active_prob, passive_prob = func([state[None, :], last_cards[None, :], lstm_state])
        # pick the most likely legal move, as A3C does when playing
        mask = get_mask(handcards, action_space, last_out_cards if mode == 1 else None)
        if mode == 0:
            mask[0] = 0
        prob = (active_prob if mode == 0 else passive_prob)[0]
        predicted = np.argmax(np.where(mask > 0, prob, -1))

        intention, r, _ = env.step_auto()
        target = action_index(to_char(intention))
        if target is not None:
            stats[mode].feed(int(predicted == target))
    return stats


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
                        stats = play_one_episode(player, self.func)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, stats)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stats = [StatCounter(), StatCounter()]

    def fetch():
        for total, episode in zip(stats, q.get()):
            if episode.count > 0:
                total.feed(episode.average)
        if verbose:
            logger.info("active accuracy: {}, passive accuracy: {}".format(*accuracies()))

    def accuracies():
        return [s.average if s.count > 0 else 0 for s in stats]

    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        fetch()
    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        fetch()
    return accuracies()


class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        nr_proc = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * nr_proc

    def _trigger(self):
        active_accuracy, passive_accuracy = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, verbose=False)
        self.trainer.monitors.put_scalar('active_accuracy', active_accuracy)
        self.trainer.monitors.put_scalar('passive_accuracy', passive_accuracy)
