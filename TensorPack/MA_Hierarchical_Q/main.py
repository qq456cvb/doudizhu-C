import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '../..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))
from tensorpack import *
from TensorPack.MA_Hierarchical_Q.expreplay import ExpReplay
from TensorPack.MA_Hierarchical_Q.env import Env
from TensorPack.MA_Hierarchical_Q.DQNModel import Model
from TensorPack.MA_Hierarchical_Q.evaluator import Evaluator
from TensorPack.MA_Hierarchical_Q.baseline_evaluator import BLEvaluator
import argparse
from env import Env as CEnv


BATCH_SIZE = 8
MAX_NUM_COMBS = 100
MAX_NUM_GROUPS = 21
ATTEN_STATE_SHAPE = 60
HIDDEN_STATE_DIM = 256 + 256 * 2 + 120
STATE_SHAPE = (MAX_NUM_COMBS, MAX_NUM_GROUPS, HIDDEN_STATE_DIM)
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 2e3
INIT_MEMORY_SIZE = MEMORY_SIZE // 10
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ  # each epoch is 100k played frames
EVAL_EPISODE = 50

NUM_ACTIONS = None
METHOD = None


class MyDataFLow(DataFlow):
    def __init__(self, exps):
        self.exps = exps

    def get_data(self):
        gens = [e.get_data() for e in self.exps]
        while True:
            batch = []
            for g in gens:
                batch.extend(next(g))
            yield batch


def get_config():
    agent_names = ['agent%d' % i for i in range(1, 4)]
    model = Model(agent_names, STATE_SHAPE, METHOD, NUM_ACTIONS, GAMMA)
    exps = [ExpReplay(
        # model=model,
        agent_name=name, player=Env(agent_names),
        state_shape=STATE_SHAPE,
        num_actions=[MAX_NUM_COMBS, MAX_NUM_GROUPS],
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.,
        update_frequency=UPDATE_FREQ
    ) for name in agent_names]

    df = MyDataFLow(exps)

    bl_evaluators = [BLEvaluator(EVAL_EPISODE, agent_names[0], 2, lambda: CEnv()),
                     BLEvaluator(EVAL_EPISODE, agent_names[1], 3, lambda: CEnv()),
                     BLEvaluator(EVAL_EPISODE, agent_names[2], 1, lambda: CEnv())]

    return AutoResumeTrainConfig(
        # always_resume=False,
        data=QueueInput(df),
        model=model,
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(model.update_target_param, verbose=True),
                every_k_steps=STEPS_PER_EPOCH // 10),    # update target network every 10k steps
            *exps,
            # ScheduledHyperParamSetter('learning_rate',
            #                           [(60, 5e-5), (100, 2e-5)]),
            *[ScheduledHyperParamSetter(
                ObjAttrParam(exp, 'exploration'),
                [(0, 1), (30, 0.5), (100, 0.3), (320, 0.1)],   # 1->0.1 in the first million steps
                interp='linear') for exp in exps],
            *bl_evaluators,
            Evaluator(EVAL_EPISODE, agent_names, lambda: Env(agent_names)),
            HumanHyperParamSetter('learning_rate'),
        ],
        # session_init=ChainInit([SaverRestore('../Hierarchical_Q/train_log/DQN-9-3-LASTCARDS/model-240000', 'agent1'),
        #                        SaverRestore('./train_log/DQN-60-MA/model-355000')]),
        # starting_epoch=0,
        # session_init=SaverRestore('train_log/DQN-54-AUG-STATE/model-75000'),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )


if __name__ == '__main__':
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
            os.path.join('train_log', 'DQN-60-MA-SELF_PLAY'))
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SimpleTrainer() if nr_gpu == 1 else AsyncMultiGPUTrainer(train_tower)
        launch_train_with_config(config, trainer)

