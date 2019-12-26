import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps, loads
import zmq
from tensorpack.utils.concurrency import LoopThread, ShareSessionThread
import queue
from simulator.tools import *


class SimulatorManager(Callback):
    class MSG_TYPE:
        SCREEN = 0
        CLICK = 1
        LOCK = 2
        UNLOCK = 3

    def __init__(self, simulators, pipe_sim2mgr, pipe_mgr2sim):
        self.sim2mgr = pipe_sim2mgr
        self.mgr2sim = pipe_mgr2sim

        self.context = zmq.Context()

        self.sim2mgr_socket = self.context.socket(zmq.PULL)
        self.sim2mgr_socket.bind(self.sim2mgr)
        self.sim2mgr_socket.set_hwm(2)

        self.mgr2sim_socket = self.context.socket(zmq.ROUTER)
        self.mgr2sim_socket.bind(self.mgr2sim)
        self.mgr2sim_socket.set_hwm(2)

        self.simulators = simulators
        for sim in self.simulators:
            ensure_proc_terminate(sim)

        self.queue = queue.Queue(maxsize=100)
        self.current_sim = None
        self.locked_sim = None

    def cxt_switch(self, sim_name):
        target_simulator = [sim for sim in self.simulators if sim.name == sim_name][0]
        if self.current_sim != target_simulator:
            self.current_sim = target_simulator
            click(self.current_sim.cxt[0], self.current_sim.cxt[1])

    def get_recv_thread(self):
        def f():
            msg = loads(self.sim2mgr_socket.recv(copy=False).bytes)
            self.queue.put(msg)

        recv_thread = LoopThread(f, pausable=False)
        # recv_thread.daemon = True
        recv_thread.name = "recv thread"
        return recv_thread

    def get_work_thread(self):
        def f():
            msg = self.queue.get()
            sim_name = msg[0]
            if msg[1] == SimulatorManager.MSG_TYPE.LOCK and self.locked_sim is None:
                self.locked_sim = sim_name
                self.mgr2sim_socket.send_multipart([sim_name.encode('utf-8'), dumps('lock')])
                time.sleep(0.2)
                return
            if self.locked_sim is not None:
                if sim_name != self.locked_sim:
                    time.sleep(0.2)
                    self.queue.put(msg)
                    return
                elif msg[1] == SimulatorManager.MSG_TYPE.UNLOCK:
                    self.locked_sim = None
                    self.mgr2sim_socket.send_multipart([sim_name.encode('utf-8'), dumps('unlock')])
                    time.sleep(0.2)
                    return

            self.cxt_switch(sim_name)
            # time.sleep(0.2)
            # print(msg[1])
            if msg[1] == SimulatorManager.MSG_TYPE.SCREEN:
                screen = grab_screen()
                self.mgr2sim_socket.send_multipart([sim_name.encode('utf-8'), dumps(screen)])
            elif msg[1] == SimulatorManager.MSG_TYPE.CLICK:
                # print('need to click')
                click(msg[2][0], msg[2][1])
                self.mgr2sim_socket.send_multipart([sim_name.encode('utf-8'), dumps('click')])

        work_thread = LoopThread(f, pausable=False)
        work_thread.name = "work thread"
        return work_thread

    def _before_train(self):
        recv_th = self.get_recv_thread()
        recv_th.start()

        work_th = self.get_work_thread()
        work_th.start()

        for sim in self.simulators:
            sim.start()
            logger.info('Simulator-{} started'.format(sim.name))

    def _after_train(self):
        for sim in self.simulators:
            sim.join()

