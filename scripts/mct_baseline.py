from tensorpack.utils.stats import StatCounter
from tqdm import tqdm
import sys
import os
if os.name == 'nt':
    sys.path.insert(0, '../build/Release')
else:
    sys.path.insert(0, '../build.linux')
from datetime import datetime
from scripts.envs import make_env
from scripts.agents import make_agent

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


if __name__ == '__main__':
    f = open('results.txt', 'w+')
    for role_id in [2, 3, 1]:
        agent = make_agent('CDQN', role_id)
        for i in range(1):
            env = make_env('MCT')
            st = StatCounter()
            for j in tqdm(range(100)):
                winning_rate = eval_episode(env, agent)
                st.feed(winning_rate)
            f.write('%s with role id %d against %s, winning rate: %f\n' % ('CDQN', role_id, 'MCT', st.average))
    f.close()
