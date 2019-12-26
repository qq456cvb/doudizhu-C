import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, '..'))
sys.path.append(ROOT_PATH)
sys.path.insert(0, os.path.join(ROOT_PATH, 'build/Release' if os.name == 'nt' else 'build'))

from tensorpack.utils.stats import StatCounter
from tqdm import tqdm

from datetime import datetime
from tensorpack.utils.utils import get_tqdm
from envs import make_env
from agents import make_agent
from multiprocessing import *


def eval_episode(env, agent):
    env.reset()
    env.prepare()
    done = False
    r = 0
    while not done:
        if env.get_role_ID() != agent.role_id:
            r, done = env.step_auto()
        else:
            r, done = env.step(agent.intention(env))
    if agent.role_id == 2:
        r = -r
    assert r != 0
    return int(r > 0)


def eval_proc(file_name):
    print(file_name)
    f = open(os.path.join('./log_more', file_name), 'w+')
    types = ['RANDOM', 'RHCP', 'CDQN', 'MCT']
    # for role_id in [2, 3, 1]:
    #     for ta in types:
    #         agent = make_agent(ta, role_id)
    #         for i in range(1):
    #             env = make_env('MCT')
    #             st = StatCounter()
    #             for j in tqdm(range(100)):
    #                 winning_rate = eval_episode(env, agent)
    #                 st.feed(winning_rate)
    #             f.write('%s with role id %d against %s, winning rate: %f\n' % (ta, role_id, 'MCT', st.average))

    for role_id in [2, 3, 1]:
        agent = make_agent('MCT', role_id)
        for i in range(1):
            for te in types:
                env = make_env(te)
                st = StatCounter()
                for j in tqdm(range(100)):
                    winning_rate = eval_episode(env, agent)
                    st.feed(winning_rate)
                f.write('%s with role id %d against %s, winning rate: %f\n' % ('MCT', role_id, te, st.average))
    f.close()


if __name__ == '__main__':
    procs = []
    for i in range(cpu_count() // 2):
        procs.append(Process(target=eval_proc, args=('res%d.txt' % i,)))
    for p in procs:
        p.start()
    for p in procs:
        p.join()
