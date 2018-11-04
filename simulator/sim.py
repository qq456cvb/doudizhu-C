
import os, sys
if os.name == 'nt':
    sys.path.insert(0, '../build/Release')
else:
    sys.path.insert(0, '../build.linux')
sys.path.insert(0, '..')
from simulator.tools import *
import ctypes
import threading

from simulator.predictor import Predictor
from card import Card
from simulator.coordinator import Coordinator
import multiprocessing
import zmq
from tensorpack.utils.serialize import dumps, loads
from env import Env as CEnv
from tensorpack.utils.stats import StatCounter
from simulator.manager import SimulatorManager
from tensorpack.utils import logger


class Simulator(multiprocessing.Process):
    class State:
        PRE_START = 0
        CALLING = 1
        PLAYING = 2
        END = 3

    templates = load_templates()
    mini_templates = load_mini_templates(templates)

    def __init__(self, idx, hwnd, pipe_sim2exps, pipe_exps2sim, pipe_sim2coord, pipe_coord2sim, pipe_sim2mgr, pipe_mgr2sim, agent_names, exploration, toggle):
        super(Simulator, self).__init__()
        self.name = 'simulator-{}'.format(idx)

        self.sim2exps = pipe_sim2exps
        self.exps2sim = pipe_exps2sim

        self.sim2coord = pipe_sim2coord
        self.coord2sim = pipe_coord2sim

        self.sim2mgr = pipe_sim2mgr
        self.mgr2sim = pipe_mgr2sim

        self.agent_names = agent_names

        # instance specific property
        self.window_rect = get_window_rect(hwnd)
        # print(self.window_rect)
        self.cxt = [10 + self.window_rect[0], 1150 + self.window_rect[1]]
        self.current_screen = None
        self.win_rates = {n: StatCounter() for n in self.agent_names}

        self.state = Simulator.State.CALLING
        self.current_lord_pos = None
        self.cached_msg = None
        self.exploration = exploration
        # for compatibility use the order, self, right, left
        self.history = [[], [], []]
        self.predictor = Predictor()
        self.toggle = toggle

    def reset_episode(self):
        self.state = Simulator.State.CALLING
        self.current_lord_pos = None
        self.cached_msg = None
        # for compatibility use the order, self, right, left
        self.history = [[], [], []]

    # def click(self, bbox):
    #     click((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2, (self.window_rect[0], self.window_rect[1]))

    def run(self):
        logger.info('simulator main loop')
        context = zmq.Context()

        sim2coord_socket = context.socket(zmq.PUSH)
        sim2coord_socket.setsockopt(zmq.IDENTITY, self.name.encode('utf-8'))
        sim2coord_socket.set_hwm(2)
        sim2coord_socket.connect(self.sim2coord)

        coord2sim_socket = context.socket(zmq.DEALER)
        coord2sim_socket.setsockopt(zmq.IDENTITY, self.name.encode('utf-8'))
        coord2sim_socket.set_hwm(2)
        coord2sim_socket.connect(self.coord2sim)

        sim2exp_sockets = []
        for sim2exp in self.sim2exps:
            sim2exp_socket = context.socket(zmq.PUSH)
            sim2exp_socket.setsockopt(zmq.IDENTITY, self.name.encode('utf-8'))
            sim2exp_socket.set_hwm(2)
            sim2exp_socket.connect(sim2exp)
            sim2exp_sockets.append(sim2exp_socket)

        sim2mgr_socket = context.socket(zmq.PUSH)
        sim2mgr_socket.setsockopt(zmq.IDENTITY, self.name.encode('utf-8'))
        sim2mgr_socket.set_hwm(2)
        sim2mgr_socket.connect(self.sim2mgr)

        mgr2sim_socket = context.socket(zmq.DEALER)
        mgr2sim_socket.setsockopt(zmq.IDENTITY, self.name.encode('utf-8'))
        mgr2sim_socket.set_hwm(2)
        mgr2sim_socket.connect(self.mgr2sim)

        # while True:
        #     time.sleep(0.3)
        #     print(self.name)
        #     sim2exp_sockets[1].send(dumps([self.name, 'haha']))

        # print('main loop')
        # while True:
        #     time.sleep(0.3)
        #     msg = loads(coord2sim_socket.recv(copy=False).bytes)
        #     print(msg)
            # sim2coord_socket.send(dumps([self.name, self.agent_names[0], np.arange(10)]))

        def request_screen():
            sim2mgr_socket.send(dumps([self.name, SimulatorManager.MSG_TYPE.SCREEN, []]))
            return loads(mgr2sim_socket.recv(copy=False).bytes)

        def request_click(bbox):
            sim2mgr_socket.send(dumps([self.name, SimulatorManager.MSG_TYPE.CLICK, [(bbox[0] + bbox[2]) // 2 + self.window_rect[0] + 6, (bbox[1] + bbox[3]) // 2 + self.window_rect[1] + 46]]))
            return loads(mgr2sim_socket.recv(copy=False).bytes)

        def request_lock():
            sim2mgr_socket.send(dumps([self.name, SimulatorManager.MSG_TYPE.LOCK, []]))
            return loads(mgr2sim_socket.recv(copy=False).bytes)

        def request_unlock():
            sim2mgr_socket.send(dumps([self.name, SimulatorManager.MSG_TYPE.UNLOCK, []]))
            return loads(mgr2sim_socket.recv(copy=False).bytes)

        def spin_lock_on_button():
            act = dict()
            while not act:
                self.current_screen = request_screen()
                cv2.imwrite('debug.png', self.current_screen)
                act = get_current_button_action(self.current_screen)
                if self.toggle.value == 0:
                    break

            return act

        def discard(act, bboxes, idxs):
            def diff(idxs, cards):
                res = []
                for i in range(len(cards)):
                    if cards[i] is not None:
                        if i in idxs:
                            res.append(i)
                    else:
                        if i not in idxs:
                            res.append(i)
                return res

            differences = diff(idxs, get_cards_bboxes(request_screen(), self.templates, bboxes=bboxes)[0])
            print(differences)
            request_lock()
            while len(differences) > 0:
                for d in differences:
                    request_click(bboxes[d])
                # request_click(bboxes[differences[0]])
                # time.sleep(0.3)
                differences = diff(idxs, get_cards_bboxes(request_screen(), self.templates, bboxes=bboxes)[0])
                print(differences)
            if 'chupai' in act:
                request_click(act['chupai'])
            elif 'alone_chupai' in act:
                request_click(act['alone_chupai'])
            elif 'ming_chupai' in act:
                request_click(act['ming_chupai'])
            request_unlock()

        game_cnt = 0
        while True:
            import psutil
            # print('memory usage is: ', psutil.virtual_memory())
            if self.toggle.value == 0:
                time.sleep(0.2)
                continue
            print('new round')
            self.current_screen = request_screen()

            act = spin_lock_on_button()
            if not act:
                continue
            print(act)
            if 'start' in act:
                request_click(act['start'])
                continue
            if self.state == Simulator.State.CALLING:
                # state has changed
                if 'reverse' in act:
                    self.state = Simulator.State.PLAYING
                    self.current_lord_pos = who_is_lord(self.current_screen)
                    while self.current_lord_pos < 0:
                        self.current_screen = request_screen()
                        self.current_lord_pos = who_is_lord(self.current_screen)
                        print('current lord pos ', self.current_lord_pos)
                        if self.toggle.value == 0:
                            break
                    continue
                if 'continuous defeat' in act:
                    request_click(act['continuous defeat'])
                    continue
                print('calling', act)
                handcards, _ = get_cards_bboxes(self.current_screen, self.templates, 0)
                cards_value, _ = CEnv.get_cards_value(Card.char2color(handcards))
                print('cards value: ', cards_value)
                assert 'jiaodizhu' in act
                request_click(act['bujiao']) if cards_value < 10 else request_click(act['jiaodizhu'])
            elif self.state == Simulator.State.PLAYING:
                if 'defeat' in act or 'victory' in act:
                    request_click(act['defeat'] if 'defeat' in act else act['victory'])
                    if self.cached_msg is None:
                        print('other player wins in one step!!!')
                        continue
                    win = is_win(self.current_screen)
                    state, action, fine_mask = self.cached_msg
                    if win:
                        sim2exp_sockets[self.current_lord_pos].send(dumps([[state, state], action, 1, True, False, [fine_mask, fine_mask]]))
                        self.win_rates[self.agent_names[self.current_lord_pos]].feed(1.)
                    else:
                        sim2exp_sockets[self.current_lord_pos].send(dumps([[state, state], action, -1, True, False, [fine_mask, fine_mask]]))
                        self.win_rates[self.agent_names[self.current_lord_pos]].feed(0.)

                    game_cnt += 1
                    if game_cnt % 100 == 0:
                        for agent in self.agent_names:
                            if self.win_rates[agent].count > 0:
                                logger.info('[last-100]{} win rate: {}'.format(agent, self.win_rates[agent].average))
                                self.win_rates[agent].reset()

                    self.reset_episode()

                    continue
                # test if we have cached msg not sent

                print('playing', act)
                left_cards, _ = get_cards_bboxes(self.current_screen, self.mini_templates, 1)
                right_cards, _ = get_cards_bboxes(self.current_screen, self.mini_templates, 2)
                if None in left_cards or None in right_cards:
                    request_click(act['buchu'])
                    time.sleep(1.)
                    continue
                assert None not in left_cards
                assert None not in right_cards
                self.history[1].extend(right_cards)
                self.history[2].extend(left_cards)
                # last_cards = left_cards
                # if not left_cards:
                #     last_cards = right_cards
                # print('last cards', last_cards)
                total_cards = np.ones([60])
                total_cards[53:56] = 0
                total_cards[57:60] = 0
                handcards, bboxes = get_cards_bboxes(self.current_screen, self.templates, 0)
                handcards = [card for card in handcards if card is not None]
                remain_cards = total_cards - Card.char2onehot60(handcards + self.history[0] + self.history[1] + self.history[2])
                print('current handcards: ', handcards)
                # left_cnt, right_cnt = get_opponent_cnts(self.current_screen, self.tiny_templates)
                # print('left cnt: ', left_cnt, 'right cnt: ', right_cnt)
                left_cnt = 17 - len(self.history[2])
                right_cnt = 17 - len(self.history[1])
                if self.current_lord_pos == 1:
                    left_cnt += 3
                if self.current_lord_pos == 2:
                    right_cnt += 3
                # assert left_cnt > 0 and right_cnt > 0
                # to be the same as C++ side, right comes before left

                right_prob_state = remain_cards * (right_cnt / (left_cnt + right_cnt))
                left_prob_state = remain_cards * (left_cnt / (left_cnt + right_cnt))
                prob_state = np.concatenate([right_prob_state, left_prob_state])
                # assert prob_state.size == 120
                # assert np.all(prob_state < 1.) and np.all(prob_state >= 0.)
                # print(prob_state)
                intention, buffer_comb, buffer_fine = self.predictor.predict(handcards, [left_cards, right_cards], prob_state, self, sim2coord_socket, coord2sim_socket)
                if self.cached_msg is not None:
                    state, action, fine_mask = self.cached_msg
                    sim2exp_sockets[self.current_lord_pos].send(
                                               dumps([[state, buffer_comb[0]], action, 0, False, False,
                                                      [fine_mask, buffer_comb[2]]]))

                    sim2exp_sockets[self.current_lord_pos].send(
                                           dumps([[buffer_comb[0], buffer_fine[0]], buffer_comb[1], 0, False, True,

                                                  [buffer_comb[2], buffer_fine[2]]]))
                self.cached_msg = buffer_fine

                self.history[0].extend(intention)
                print('intention is: ', intention)
                intention.sort(key=lambda k: Card.cards_to_value[k])
                if len(intention) == 0:
                    request_click(act['buchu'])
                else:
                    i = 0
                    j = 0
                    to_click = []
                    to_click_idxs = []
                    while j < len(intention):
                        if handcards[i] == intention[j]:
                            to_click_idxs.append(i)
                            to_click.append(bboxes[i])
                            i += 1
                            j += 1
                        else:
                            i += 1
                    for bbox in to_click:
                        request_click(bbox)
                    time.sleep(0.5)
                    request_click([1310, 760, 1310, 760])
            time.sleep(1.)


if __name__ == '__main__':
    a = 3
    b = 4
    c = 4
    res = set()
    for x in range(1, a + 1):
        for y in range(1, b + 1):
            for z in range(1, c + 1):
                if x + y > z and x + z > y and y + z > x:
                    res.add((x, y, z))
    print(len(res))