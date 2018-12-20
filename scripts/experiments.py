from tensorpack.utils.stats import StatCounter
import sys
import os
if os.name == 'nt':
    sys.path.insert(0, '../build/Release')
else:
    sys.path.insert(0, '../build.linux')
from datetime import datetime
from scripts.envs import make_env
from scripts.agents import make_agent


types = ['RANDOM', 'HCWB']


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
    for te in types:
        for ta in types:
            for role_id in [1, 2, 3]:
                agent = make_agent(ta, role_id)
                for i in range(1):
                    env = make_env(te)
                    st = StatCounter()
                    for j in range(100):
                        winning_rate = eval_episode(env, agent)
                        st.feed(winning_rate)
                        print('one episode')
                    print('%s with role id %d against %s, winning rate: %f'.format(ta, role_id, te, st.average))

