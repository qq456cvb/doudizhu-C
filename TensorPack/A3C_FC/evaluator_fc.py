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
from utils import inference_minor_cards, gputimeblock, give_cards_without_minor, pick_main_cards


def play_one_episode(env, func):
    def take_action_from_prob(prob, mask):
        prob = prob[0]
        # to avoid numeric difficulty
        prob[mask == 0] = -1
        return np.argmax(prob)

    # return char minor cards output
    def inference_minor_util60(role_id, handcards, num, is_pair, dup_mask, main_cards_char):
        for main_card in main_cards_char:
            handcards.remove(main_card)

        s = get_mask(handcards, action_space, None).astype(np.float32)
        outputs = []
        minor_type = 1 if is_pair else 0
        for i in range(num):
            input_single, input_pair, _, _ = get_masks(handcards, None)
            _, _, _, _, _, _, minor_response_prob = func(
                [np.array([role_id]), s.reshape(1, -1), np.zeros([1, 9085]), np.array([minor_type])]
            )

            # give minor cards
            mask = None
            if is_pair:
                mask = np.concatenate([input_pair, [0, 0]]) * dup_mask
            else:
                mask = input_single * dup_mask

            minor_response = take_action_from_prob(minor_response_prob, mask)
            dup_mask[minor_response] = 0

            # convert network output to char cards
            handcards.remove(to_char(minor_response + 3))
            if is_pair:
                handcards.remove(to_char(minor_response + 3))
            s = get_mask(handcards, action_space, None).astype(np.float32)

            # save to output
            outputs.append(to_char(minor_response + 3))
            if is_pair:
                outputs.append(to_char(minor_response + 3))
        return outputs

    def inference_minor_cards60(role_id, category, s, handcards, seq_length, dup_mask, main_cards_char):
        if category == Category.THREE_ONE.value:
            return inference_minor_util60(role_id, handcards, 1, False, dup_mask, main_cards_char)
        if category == Category.THREE_TWO.value:
            return inference_minor_util60(role_id, handcards, 1, True, dup_mask, main_cards_char)
        if category == Category.THREE_ONE_LINE.value:
            return inference_minor_util60(role_id, handcards, seq_length, False, dup_mask, main_cards_char)
        if category == Category.THREE_TWO_LINE.value:
            return inference_minor_util60(role_id, handcards, seq_length, True, dup_mask, main_cards_char)
        if category == Category.FOUR_TWO.value:
            return inference_minor_util60(role_id, handcards, 2, False, dup_mask, main_cards_char)

    env.reset()
    init_cards = np.arange(21)
    # init_cards = np.append(init_cards[::4], init_cards[1::4])
    env.prepare_manual(init_cards)
    r = 0
    while r == 0:
        last_cards_value = env.get_last_outcards()
        last_cards_char = to_char(last_cards_value)
        last_out_cards = Card.val2onehot60(last_cards_value)
        last_category_idx = env.get_last_outcategory_idx()
        curr_cards_char = to_char(env.get_curr_handcards())
        is_active = True if last_cards_value.size == 0 else False

        s = get_mask(curr_cards_char, action_space, None if is_active else last_cards_char).astype(np.float32)
        last_state = get_mask(last_cards_char, action_space, None).astype(np.float32)
        # print(s.shape)

        role_id = env.get_role_ID()
        # print('%s current cards' % ('lord' if role_id == 2 else 'farmer'), curr_cards_char)

        intention = None
        if role_id == 2:
            if is_active:

                # first get mask
                decision_mask, response_mask, _, length_mask = get_mask_alter(curr_cards_char, [], last_category_idx)

                _, _, _, active_decision_prob, active_response_prob, active_seq_prob, _ = func(
                    [np.array([role_id]), s.reshape(1, -1), np.zeros([1, 9085]), np.zeros([s.shape[0]])]
                )

                # make decision depending on output
                active_decision = take_action_from_prob(active_decision_prob, decision_mask)

                active_category_idx = active_decision + 1

                # get response
                active_response = take_action_from_prob(active_response_prob, response_mask[active_decision])

                seq_length = 0
                # next sequence length
                if active_category_idx == Category.SINGLE_LINE.value or \
                        active_category_idx == Category.DOUBLE_LINE.value or \
                        active_category_idx == Category.TRIPLE_LINE.value or \
                        active_category_idx == Category.THREE_ONE_LINE.value or \
                        active_category_idx == Category.THREE_TWO_LINE.value:
                    seq_length = take_action_from_prob(active_seq_prob, length_mask[active_decision][active_response]) + 1

                # give main cards
                intention = give_cards_without_minor(active_response, last_cards_value, active_category_idx, seq_length)

                # then give minor cards
                if active_category_idx == Category.THREE_ONE.value or \
                        active_category_idx == Category.THREE_TWO.value or \
                        active_category_idx == Category.THREE_ONE_LINE.value or \
                        active_category_idx == Category.THREE_TWO_LINE.value or \
                        active_category_idx == Category.FOUR_TWO.value:
                    dup_mask = np.ones([15])
                    if seq_length > 0:
                        for i in range(seq_length):
                            dup_mask[intention[0] - 3 + i] = 0
                    else:
                        dup_mask[intention[0] - 3] = 0
                    intention = np.concatenate([intention,
                                                to_value(inference_minor_cards60(role_id, active_category_idx, s.copy(),
                                                                                 curr_cards_char.copy(), seq_length,
                                                                                 dup_mask, to_char(intention)))])
            else:
                # print(to_char(last_cards_value), is_bomb, last_category_idx)
                decision_mask, response_mask, bomb_mask, _ = get_mask_alter(curr_cards_char, to_char(last_cards_value),
                                                                            last_category_idx)

                passive_decision_prob, passive_bomb_prob, passive_response_prob, _, _, _, _ = func(
                    [np.array([role_id]), s.reshape(1, -1), last_state.reshape(1, -1), np.zeros([s.shape[0]])])

                passive_decision = take_action_from_prob(passive_decision_prob, decision_mask)

                if passive_decision == 0:
                    intention = np.array([])
                elif passive_decision == 1:

                    passive_bomb = take_action_from_prob(passive_bomb_prob, bomb_mask)

                    # converting 0-based index to 3-based value
                    intention = np.array([passive_bomb + 3] * 4)

                elif passive_decision == 2:
                    intention = np.array([16, 17])
                elif passive_decision == 3:
                    passive_response = take_action_from_prob(passive_response_prob, response_mask)

                    intention = give_cards_without_minor(passive_response, last_cards_value, last_category_idx, None)
                    if last_category_idx == Category.THREE_ONE.value or \
                            last_category_idx == Category.THREE_TWO.value or \
                            last_category_idx == Category.THREE_ONE_LINE.value or \
                            last_category_idx == Category.THREE_TWO_LINE.value or \
                            last_category_idx == Category.FOUR_TWO.value:
                        dup_mask = np.ones([15])
                        seq_length = get_seq_length(last_category_idx, last_cards_value)
                        if seq_length:
                            for i in range(seq_length):
                                dup_mask[intention[0] - 3 + i] = 0
                        else:
                            dup_mask[intention[0] - 3] = 0
                        intention = np.concatenate([intention,
                                                    to_value(inference_minor_cards60(role_id, last_category_idx, s.copy(),
                                                                                     curr_cards_char.copy(), seq_length,
                                                                                     dup_mask, to_char(intention)))])
            # since step auto needs full last card group info, we do not explicitly feed card type
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
        # if t > 10 * 60:  # eval takes too long
        #     self.eval_episode = int(self.eval_episode * 0.94)

    def _trigger_epoch(self):
        t = time.time()
        farmer_win_rate = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, verbose=False)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('farmer win rate', farmer_win_rate)
        self.trainer.monitors.put_scalar('lord win rate', 1 - farmer_win_rate)


if __name__ == '__main__':
    env = Env()
    stat = StatCounter()
    init_cards = np.arange(15)
    # init_cards = np.append(init_cards[::4], init_cards[1::4])
    for _ in range(1000):
        env.reset()
        env.prepare_manual(init_cards)
        r = 0
        while r == 0:
            _, r, _ = env.step_auto()
        stat.feed(int(r < 0))
    print('lord win rate: {}'.format(stat.average))
