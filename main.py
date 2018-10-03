#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""
import sys
sys.path.append('../build')
from env import Env as CEnv
from utils import to_char

if __name__ == '__main__':
    env = CEnv()
    rounds = 10
    for _ in range(rounds):
        env.reset()
        env.prepare()
        r = 0
        count = 0
        round_count = 1
        while r == 0:
            intention, r, cate, idx = env.step_auto()
            if r != 0 and idx == 0:
                count += 1
            print('player{}: puts:{} '.format(idx, intention))
            round_count += 1
            if round_count % 3 == 0:
                round_count = 1
        print('done')
    print('winning rate: ', count / rounds)