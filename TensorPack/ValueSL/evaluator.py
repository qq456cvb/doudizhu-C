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
from logger import Logger
from utils import to_char
from card import Card, action_space, Category
import numpy as np
from utils import get_mask, get_minor_cards, train_fake_action_60, get_masks, test_fake_action
from utils import get_seq_length, pick_minor_targets, to_char, to_value, get_mask_alter, discard_onehot_from_s_60
from utils import inference_minor_cards, gputimeblock, scheduled_run, give_cards_without_minor, pick_main_cards


def play_one_episode(env, func):
    env.reset()
    env.prepare()
    r = 0
    while r == 0:
        s = env.get_state_all_cards()
        intention, r, category_idx = env.step_auto()

        if category_idx == 14:
            continue
        minor_cards_targets = pick_minor_targets(category_idx, to_char(intention))

        if not is_active:
            if category_idx == Category.QUADRIC.value and category_idx != last_category_idx:
                passive_decision_input = 1
                passive_bomb_input = intention[0] - 3
                passive_decision_prob, passive_bomb_prob, _, _, _, _, _ = func(
                    [s.reshape(1, -1), last_out_cards.reshape(1, -1), np.zeros([s.shape[0]])])
                stats[0].feed(int(passive_decision_input == np.argmax(passive_decision_prob)))
                stats[1].feed(int(passive_bomb_input == np.argmax(passive_bomb_prob)))

            else:
                if category_idx == Category.BIGBANG.value:
                    passive_decision_input = 2
                    passive_decision_prob, _, _, _, _, _, _ = func(
                        [s.reshape(1, -1), last_out_cards.reshape(1, -1), np.zeros([s.shape[0]])])
                    stats[0].feed(int(passive_decision_input == np.argmax(passive_decision_prob)))
                else:
                    if category_idx != Category.EMPTY.value:
                        passive_decision_input = 3
                        # OFFSET_ONE
                        # 1st, Feb - remove relative card output since shift is hard for the network to learn
                        passive_response_input = intention[0] - 3
                        if passive_response_input < 0:
                            print("something bad happens")
                            passive_response_input = 0
                        passive_decision_prob, _, passive_response_prob, _, _, _, _ = func(
                            [s.reshape(1, -1), last_out_cards.reshape(1, -1), np.zeros([s.shape[0]])])
                        stats[0].feed(int(passive_decision_input == np.argmax(passive_decision_prob)))
                        stats[2].feed(int(passive_response_input == np.argmax(passive_response_prob)))
                    else:
                        passive_decision_input = 0
                        passive_decision_prob, _, _, _, _, _, _ = func(
                            [s.reshape(1, -1), last_out_cards.reshape(1, -1), np.zeros([s.shape[0]])])
                        stats[0].feed(int(passive_decision_input == np.argmax(passive_decision_prob)))

        else:
            seq_length = get_seq_length(category_idx, intention)

            # ACTIVE OFFSET ONE!
            active_decision_input = category_idx - 1
            active_response_input = intention[0] - 3
            _, _, _, active_decision_prob, active_response_prob, active_seq_prob, _ = func(
                [s.reshape(1, -1), last_out_cards.reshape(1, -1), np.zeros([s.shape[0]])]
            )

            stats[3].feed(int(active_decision_input == np.argmax(active_decision_prob)))
            stats[4].feed(int(active_response_input == np.argmax(active_response_prob)))

            if seq_length is not None:
                # length offset one
                seq_length_input = seq_length - 1
                stats[5].feed(int(seq_length_input == np.argmax(active_seq_prob)))

        if minor_cards_targets is not None:
            main_cards = pick_main_cards(category_idx, to_char(intention))
            handcards = curr_cards_char.copy()
            state = s.copy()
            for main_card in main_cards:
                handcards.remove(main_card)
            cards_onehot = Card.char2onehot60(main_cards)

            # we must make the order in each 4 batch correct...
            discard_onehot_from_s_60(state, cards_onehot)

            is_pair = False
            minor_type = 0
            if category_idx == Category.THREE_TWO.value or category_idx == Category.THREE_TWO_LINE.value:
                is_pair = True
                minor_type = 1
            for target in minor_cards_targets:
                target_val = Card.char2value_3_17(target) - 3
                _, _, _, _, _, _, minor_response_prob = func(
                    [state.copy().reshape(1, -1), last_out_cards.reshape(1, -1), np.array([minor_type])]
                )
                stats[6].feed(int(target_val == np.argmax(minor_response_prob)))
                cards = [target]
                handcards.remove(target)
                if is_pair:
                    if target not in handcards:
                        logger.warn('something wrong...')
                        logger.warn('minor', target)
                        logger.warn('main_cards', main_cards)
                        logger.warn('handcards', handcards)
                    else:
                        handcards.remove(target)
                        cards.append(target)

                # correct for one-hot state
                cards_onehot = Card.char2onehot60(cards)

                # print(s.shape)
                # print(cards_onehot.shape)
                discard_onehot_from_s_60(state, cards_onehot)
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
                    scores = [stat.average if stat.count > 0 else -1 for stat in stats]
                    self.queue_put_stoppable(self.q, scores)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stats = [StatCounter() for _ in range(7)]

    def fetch():
        scores = q.get()
        for i, score in enumerate(scores):
            if scores[i] >= 0:
                stats[i].feed(scores[i])
        accs = [stat.average if stat.count > 0 else 0 for stat in stats]
        if verbose:
            logger.info("passive decision accuracy: {}\n"
                        "passive bomb accuracy: {}\n"
                        "passive response accuracy: {}\n"
                        "active decision accuracy: {}\n"
                        "active response accuracy: {}\n"
                        "active sequence accuracy: {}\n"
                        "minor response accuracy: {}\n".format(accs[0], accs[1], accs[2], accs[3], accs[4], accs[5], accs[6]))

    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        fetch()
    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        fetch()
    accs = [stat.average if stat.count > 0 else 0 for stat in stats]
    return accs


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
        t = time.time()
        accs = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, verbose=False)
        t = time.time() - t
        # if t > 10 * 60:  # eval takes too long
        #     self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('passive_decision_accuracy', accs[0])
        self.trainer.monitors.put_scalar('passive_bomb_accuracy', accs[1])
        self.trainer.monitors.put_scalar('passive_response_accuracy', accs[2])
        self.trainer.monitors.put_scalar('active_decision_accuracy', accs[3])
        self.trainer.monitors.put_scalar('active_response_accuracy', accs[4])
        self.trainer.monitors.put_scalar('active_sequence_accuracy', accs[5])
        self.trainer.monitors.put_scalar('minor_response_accuracy', accs[6])