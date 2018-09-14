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
from TensorPack.MA_Hierarchical_Q.DQNModel import Model
from TensorPack.MA_Hierarchical_Q.env import Env
from simulator.expreplay import ExpReplay
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
import six
from abc import abstractmethod, ABCMeta
from tensorpack import *
from tensorpack.utils.stats import StatCounter
import argparse

toggle = False


BATCH_SIZE = 8
MAX_NUM_COMBS = 100
MAX_NUM_GROUPS = 21
ATTEN_STATE_SHAPE = 60
HIDDEN_STATE_DIM = 256 + 256 + 120
STATE_SHAPE = (MAX_NUM_COMBS, MAX_NUM_GROUPS, HIDDEN_STATE_DIM)
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 3e3
INIT_MEMORY_SIZE = MEMORY_SIZE // 10
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ  # each epoch is 100k played frames
EVAL_EPISODE = 100

NUM_ACTIONS = None
METHOD = None


def esc_pressed():
    global toggle
    toggle = not toggle
    print("toggle changed to ", toggle)


class Simulator(multiprocessing.Process):
    class State:
        PRE_START = 0
        CALLING = 1
        PLAYING = 2
        END = 3

    def __init__(self, name, pipe_c2s, pipe_s2c, pipe_sim2coord, pipe_coord2sim, agent_names, exploration):
        super(Simulator, self).__init__()
        self.name = name
        self.c2s = pipe_c2s
        self.s2c = pipe_s2c
        self.sim2coord = pipe_sim2coord
        self.coord2sim = pipe_coord2sim
        self.agent_names = agent_names
        self.templates = load_templates()
        self.mini_templates = dict()
        for t in self.templates:
            if t == 'Joker':
                self.mini_templates[t] = cv2.imread('./templates/Joker_mini.png', cv2.IMREAD_GRAYSCALE)
            else:
                self.mini_templates[t] = cv2.resize(self.templates[t], (0, 0), fx=0.7, fy=0.7)
        self.tiny_templates = load_tiny_templates()
        # self.grab_screen = grab_screen()

        # instance specific property
        self.window_rect = get_window_rect()
        self.current_screen = None
        self.win_rates = {n: StatCounter() for n in self.agent_names}

        self.state = Simulator.State.CALLING
        self.current_lord_pos = None
        self.cached_msg = None
        self.exploration = exploration
        # for compatibility use the order, self, right, left
        self.history = [[], [], []]
        self.predictor = Predictor()

    def reset_episode(self):
        self.state = Simulator.State.CALLING
        self.current_lord_pos = None
        self.cached_msg = None
        # for compatibility use the order, self, right, left
        self.history = [[], [], []]

    def spin_lock_on_button(self):
        act = dict()
        while not act:
            time.sleep(0.2)
            self.current_screen = grab_screen()
            cv2.imwrite('debug.png', self.current_screen)
            act = get_current_button_action(self.current_screen)

        return act

    def click(self, bbox):
        click((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2, (self.window_rect[0], self.window_rect[1]))

    def discard(self, act, bboxes, idxs):
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

        differences = diff(idxs, get_cards_bboxes(grab_screen(), self.templates)[0])
        while len(differences) > 0:
            for d in differences:
                self.click(bboxes[d])
                time.sleep(0.1)
            time.sleep(0.75)
            differences = diff(idxs, get_cards_bboxes(grab_screen(), self.templates)[0])
        if 'chupai' in act:
            self.click(act['chupai'])
        elif 'alone_chupai' in act:
            self.click(act['alone_chupai'])
        elif 'ming_chupai' in act:
            self.click(act['ming_chupai'])

    def main_loop(self):
        context = zmq.Context()

        sim2coord_socket = context.socket(zmq.PUSH)
        sim2coord_socket.setsockopt(zmq.IDENTITY, self.name.encode('utf-8'))
        sim2coord_socket.set_hwm(20)
        sim2coord_socket.connect(self.sim2coord)

        coord2sim_socket = context.socket(zmq.DEALER)
        coord2sim_socket.setsockopt(zmq.IDENTITY, self.name.encode('utf-8'))
        coord2sim_socket.set_hwm(20)
        coord2sim_socket.connect(self.coord2sim)

        c2s_socket = context.socket(zmq.PULL)
        c2s_socket.connect(self.c2s)
        c2s_socket.set_hwm(20)
        s2c_socket = context.socket(zmq.ROUTER)
        s2c_socket.connect(self.s2c)
        s2c_socket.set_hwm(20)

        global toggle
        while True:
            if not toggle:
                time.sleep(0.2)
                continue
            self.current_screen = grab_screen()

            act = self.spin_lock_on_button()
            print(act)
            if self.state == Simulator.State.CALLING:
                # state has changed
                if 'chupai' in act or 'alone_chupai' in act or 'ming_chupai' in act or 'yaobuqi' in act:
                    self.state = Simulator.State.PLAYING
                    self.current_lord_pos = who_is_lord(self.current_screen)
                    assert self.current_lord_pos >= 0
                    continue
                print('calling', act)
                handcards, _ = get_cards_bboxes(self.current_screen, self.templates, 0)
                cards_value, _ = Env.get_cards_value(Card.char2color(handcards))
                print('cards value: ', cards_value)
                if 'qiangdizhu' in act:
                    self.click(act['buqiang']) if cards_value < 10 else self.click(act['qiangdizhu'])
                else:
                    if 'bujiabei' in act:
                        self.click(act['bujiabei'])
                        time.sleep(1.)
                        continue
                    # assert 'jiaodizhu' in act
                    self.click(act['bujiao']) if cards_value < 10 else self.click(act['jiaodizhu'])
            elif self.state == Simulator.State.PLAYING:
                if 'end' in act or 'continous_end' in act:
                    time.sleep(1.)
                    self.click(act['end'] if 'end' in act else act['continous_end'])
                    time.sleep(1.)
                    if self.cached_msg is None:
                        print('other player wins in one step!!!')
                        continue
                    win = True
                    state, action, fine_mask = self.cached_msg
                    if win:
                        s2c_socket.send_multipart([self.agent_names[self.current_lord_pos].encode('utf-8'), dumps([[state, state], action, 1, True, False, [fine_mask, fine_mask]])])
                        self.win_rates[self.agent_names[self.current_lord_pos]].feed(1.)
                    else:
                        s2c_socket.send_multipart([self.agent_names[self.current_lord_pos].encode('utf-8'), dumps([[state, state], action, -1, True, False, [fine_mask, fine_mask]])])
                        self.win_rates[self.agent_names[self.current_lord_pos]].feed(0.)

                    self.reset_episode()
                    continue
                # test if we have cached msg not sent

                print('playing', act)
                left_cards, _ = get_cards_bboxes(self.current_screen, self.mini_templates, 1)
                right_cards, _ = get_cards_bboxes(self.current_screen, self.mini_templates, 2)

                assert None not in left_cards
                assert None not in right_cards
                self.history[1].extend(right_cards)
                self.history[2].extend(left_cards)
                last_cards = left_cards
                if not left_cards:
                    last_cards = right_cards
                print('last cards', last_cards)
                total_cards = np.ones([60])
                total_cards[53:56] = 0
                total_cards[57:60] = 0
                handcards, bboxes = get_cards_bboxes(self.current_screen, self.templates, 0)
                remain_cards = total_cards - Card.char2onehot60(handcards + self.history[0] + self.history[1] + self.history[2])
                print('current handcards: ', handcards)
                left_cnt, right_cnt = get_opponent_cnts(self.current_screen, self.tiny_templates)
                print('left cnt: ', left_cnt, 'right cnt: ', right_cnt)
                assert left_cnt > 0 and right_cnt > 0
                # to be the same as C++ side, right comes before left

                right_prob_state = remain_cards * (right_cnt / (left_cnt + right_cnt))
                left_prob_state = remain_cards * (left_cnt / (left_cnt + right_cnt))
                prob_state = np.concatenate([right_prob_state, left_prob_state])
                assert prob_state.size == 120
                assert np.all(prob_state < 1.) and np.all(prob_state >= 0.)
                # print(prob_state)
                intention, buffer_comb, buffer_fine = self.predictor.predict(handcards, last_cards, prob_state, self, coord2sim_socket, sim2coord_socket)
                if self.cached_msg is not None:
                    state, action, fine_mask = self.cached_msg
                    s2c_socket.send_multipart([self.agent_names[self.current_lord_pos].encode('utf-8'),
                                               dumps([[state, buffer_comb[0]], action, 0, False, False,
                                                      [fine_mask, buffer_comb[2]]])])

                s2c_socket.send_multipart([self.agent_names[self.current_lord_pos].encode('utf-8'),
                                           dumps([[buffer_comb[0], buffer_fine[0]], buffer_comb[1], 0, False, True,
                                                  [buffer_comb[2], buffer_fine[2]]])])
                self.cached_msg = buffer_fine

                self.history[0].extend(intention)
                print('intention is: ', intention)
                intention.sort(key=lambda k: Card.cards_to_value[k], reverse=True)
                if len(intention) == 0:
                    self.click(act['buchu']) if 'buchu' in act else self.click(act['yaobuqi'])
                else:
                    # i = 0
                    # j = 0
                    # to_click = []
                    # to_click_idxs = []
                    # while j < len(intention):
                    #     if handcards[i] == intention[j]:
                    #         to_click_idxs.append(i)
                    #         to_click.append(bboxes[i])
                    #         i += 1
                    #         j += 1
                    #     else:
                    #         i += 1
                    self.discard(act, bboxes, [])
            time.sleep(1.)


def hook():
    ctypes.windll.user32.RegisterHotKey(None, 1, 0, win32con.VK_ESCAPE)

    try:
        msg = ctypes.wintypes.MSG()
        while ctypes.windll.user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == win32con.WM_HOTKEY:
                esc_pressed()
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageA(ctypes.byref(msg))
    finally:
        ctypes.windll.user32.UnregisterHotKey(None, 1)


class SimulatorManager(Callback):
    def __init__(self, simulators):
        self.simulators = simulators
        for sim in self.simulators:
            ensure_proc_terminate(sim)

    def _before_train(self):
        for sim in self.simulators:
            sim.start()

    def _after_train(self):
        for sim in self.simulators:
            sim.join()

# class RandomClient(multiprocessing.Process):
#     def __init__(self, pipe_c2s, pipe_s2c):
#         super(RandomClient, self).__init__()
#         self.c2s = pipe_c2s
#         self.s2c = pipe_s2c
#
#     def run(self):
#         context = zmq.Context()
#
#         c2s_socket = context.socket(zmq.PULL)
#         c2s_socket.bind(self.c2s)
#         c2s_socket.set_hwm(20)
#
#         s2c_socket = context.socket(zmq.ROUTER)
#         s2c_socket.connect(self.s2c)
#         s2c_socket.set_hwm(20)
#         print('run')
#         agent_name = c2s_socket.recv(copy=False).bytes
#         print(agent_name)
#         while True:
#             msg = dumps((b'test'))
#             s2c_socket.send_multipart([agent_name, msg])
#             # _, msg = loads(c2s_socket.recv(copy=False).bytes)
#             # print('server received: ', msg)


class MyDataFLow(DataFlow):
    def __init__(self, exps):
        self.exps = exps

    def get_data(self):
        gens = [e.get_data() for e in self.exps]
        while True:
            batch = []
            # TODO: balance between batch generation?
            for g in gens:
                batch.extend(next(g))
            yield batch


if __name__ == '__main__':
    namec2s = 'tcp://127.0.0.1:8888'
    names2c = 'tcp://127.0.0.1:9999'
    name_sim2coord = 'tcp://127.0.0.1:6666'
    name_coord2sim = 'tcp://127.0.0.1:7777'

    agent_names = ['agent%d' % i for i in range(1, 4)]

    sim = Simulator(name='simulator-1', pipe_c2s=namec2s, pipe_s2c=names2c, pipe_sim2coord=name_sim2coord, pipe_coord2sim=name_coord2sim, agent_names=agent_names, exploration=0.05)
    manager = SimulatorManager([sim])

    coordinator = Coordinator(agent_names, sim2coord=name_sim2coord, coord2sim=name_coord2sim)
    hotkey_t = threading.Thread(target=hook, args=())

    def get_config():

        model = Model(agent_names, STATE_SHAPE, METHOD, NUM_ACTIONS, GAMMA)
        exps = [ExpReplay(
            agent_name=name,
            state_shape=STATE_SHAPE,
            num_actions=[MAX_NUM_COMBS, MAX_NUM_GROUPS],
            batch_size=BATCH_SIZE,
            memory_size=MEMORY_SIZE,
            init_memory_size=INIT_MEMORY_SIZE,
            init_exploration=1.,
            update_frequency=UPDATE_FREQ,
            pipe_c2s=namec2s,
            pipe_s2c=names2c
        ) for name in agent_names]

        df = MyDataFLow(exps)

        return AutoResumeTrainConfig(
            # always_resume=False,
            data=QueueInput(df),
            model=model,
            callbacks=[
                ModelSaver(),
                PeriodicTrigger(
                    RunOp(model.update_target_param, verbose=True),
                    every_k_steps=STEPS_PER_EPOCH // 10),  # update target network every 10k steps
                # the following order is important
                coordinator,
                manager,
                *exps,
                # ScheduledHyperParamSetter('learning_rate',
                #                           [(60, 5e-5), (100, 2e-5)]),
                # *[ScheduledHyperParamSetter(
                #     ObjAttrParam(exp, 'exploration'),
                #     [(0, 1), (30, 0.5), (100, 0.3), (320, 0.1)],  # 1->0.1 in the first million steps
                #     interp='linear') for exp in exps],
                # Evaluator(EVAL_EPISODE, agent_names, lambda: Env(agent_names)),
                HumanHyperParamSetter('learning_rate'),
            ],
            session_init=ChainInit(
                [SaverRestore('../Hierarchical_Q/train_log/DQN-9-3-LASTCARDS/model-240000', 'agent1'),
                 SaverRestore('./train_log/DQN-60-MA/model-355000')]),
            # starting_epoch=0,
            # session_init=SaverRestore('train_log/DQN-54-AUG-STATE/model-75000'),
            steps_per_epoch=STEPS_PER_EPOCH,
            max_epoch=1000,
        )
    # client = RandomClient(pipe_c2s=namec2s, pipe_s2c=names2c)
    # # client.run()
    # ensure_proc_terminate(client)
    # # start_proc_mask_signal(client)
    # client.start()
    # client.join()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling'], default='Double')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    METHOD = args.algo
    # set num_actions
    NUM_ACTIONS = max(MAX_NUM_GROUPS, MAX_NUM_COMBS)

    nr_gpu = get_nr_gpu()
    train_tower = list(range(nr_gpu))
    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state', 'comb_mask'],
            output_names=['Qvalue']))
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'DQN-REALDATA'))
        config = get_config()
        hotkey_t.start()
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SimpleTrainer() if nr_gpu == 1 else AsyncMultiGPUTrainer(train_tower)
        launch_train_with_config(config, trainer)

        hotkey_t.join()


