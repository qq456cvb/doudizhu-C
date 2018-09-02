from tensorpack import *
from TensorPack.MA_Hierarchical_Q.expreplay import ExpReplay
from TensorPack.MA_Hierarchical_Q.env import Env
from TensorPack.MA_Hierarchical_Q.DQNModel import Model
import argparse
import os


BATCH_SIZE = 8
MAX_NUM_COMBS = 100
MAX_NUM_GROUPS = 21
ATTEN_STATE_SHAPE = 60
HIDDEN_STATE_DIM = 256 + 120
STATE_SHAPE = (MAX_NUM_COMBS, MAX_NUM_GROUPS, HIDDEN_STATE_DIM)
ACTION_REPEAT = 4   # aka FRAME_SKIP
UPDATE_FREQ = 4

GAMMA = 0.99

MEMORY_SIZE = 1e2
INIT_MEMORY_SIZE = MEMORY_SIZE // 20
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ  # each epoch is 100k played frames
EVAL_EPISODE = 100

NUM_ACTIONS = None
METHOD = None


class MyDataFLow(DataFlow):
    def __init__(self, exps):
        self.exps = exps

    def get_data(self):
        while True:
            batch = []
            for e in self.exps:
                batch.extend(next(e.get_data()))
            yield batch

# class Agent:
#     def __init__(self, model, exp):
#         self.model = model
#         self.exp = exp


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
                [(0, 1), (30, 0.5), (60, 0.1), (320, 0.01)],   # 1->0.1 in the first million steps
                interp='linear') for exp in exps],
            # Evaluator(
            #     EVAL_EPISODE, ['state', 'comb_mask', 'fine_mask'], ['Qvalue'], [MAX_NUM_COMBS, MAX_NUM_GROUPS], get_player),
            HumanHyperParamSetter('learning_rate'),
        ],
        # starting_epoch=30,
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
            os.path.join('train_log', 'DQN-60-MA'))
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SimpleTrainer() if nr_gpu == 1 else AsyncMultiGPUTrainer(train_tower)
        launch_train_with_config(config, trainer)

