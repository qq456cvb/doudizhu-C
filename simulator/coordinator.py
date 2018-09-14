from tensorpack import *
import zmq
from tensorpack.utils.concurrency import LoopThread
from tensorpack.utils.serialize import loads, dumps
import six
if six.PY3:
    from concurrent import futures

    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

PREDICTOR_THREAD = 4


class Coordinator(Callback):
    def __init__(self, agent_names, sim2coord, coord2sim):
        self.agent_names = agent_names
        self.context = zmq.Context()
        self.sim2coord_socket = self.context.socket(zmq.PULL)
        self.sim2coord_socket.set_hwm(20)
        self.sim2coord_socket.connect(sim2coord)

        self.coord2sim_socket = self.context.socket(zmq.ROUTER)
        self.coord2sim_socket.set_hwm(20)
        self.coord2sim_socket.connect(coord2sim)

    def _setup_graph(self):
        self.predictors = {n: MultiThreadAsyncPredictor([self.trainer.get_predictor([n + '/state:0', n + '_comb_mask:0', n + '/fine_mask:0'], [n + '/Qvalue:0']) for _ in range(PREDICTOR_THREAD)]) for n in self.player.get_all_agent_names()}

    def _before_train(self):
        for p in self.predictors:
            p.start()

        def f():
            msg = loads(self.sim2coord_socket.recv(copy=False).bytes)
            sim_name = msg[0]
            agent_name = msg[1]

            def cb(outputs):
                try:
                    output = outputs.result()
                except CancelledError:
                    logger.info("{} cancelled.".format(sim_name))
                    return
                self.coord2sim_socket.send_multipart([sim_name, dumps(output[0])])
            self.predictors[agent_name].put_task(msg[2:], cb)

        self.recv_thread = LoopThread(f)
        self.recv_thread.daemon = True
        self.recv_thread.start()
