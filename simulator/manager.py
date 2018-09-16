
import os, sys
if os.name == 'nt':
    sys.path.insert(0, '../build/Release')
else:
    sys.path.insert(0, '../build.linux')
sys.path.insert(0, '..')
from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal


class SimulatorManager(Callback):
    def __init__(self, simulators):
        self.simulators = simulators
        for sim in self.simulators:
            ensure_proc_terminate(sim)

    def _before_train(self):
        for sim in self.simulators:
            sim.start()
            logger.info('Simulator-{} started'.format(sim.name))

    def _after_train(self):
        for sim in self.simulators:
            sim.join()
