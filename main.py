#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""
import sys
sys.path.append('./build')
from env import Env as CEnv
from utils import to_char

if __name__ == '__main__':
    env = CEnv()
    rounds = 500
    count = 0
    for this_round in range(rounds):
        env.reset()
        env.prepare()
        r = 0
        round_count = 1
        print('-' * 50 + 'Game start at rounds {}!'.format(this_round) + '-' * 50)
        while r == 0:
            intention, r, cate, idx, p1, p2, idx1, p0, idx2 = env.step_auto()
            if (r != 0) and (idx == 0):
                count += 1
            print('player{}: puts:{} '.format(idx, intention))
            print('card type is ', cate)
            print('my_player hand cards: {}'.format(p1))
            print('player{} hand cards: {}'.format(idx1, p2))
            print('player{} hand cards: {}'.format(idx2, p0))
            print('-' * 50)
            round_count += 1
            if round_count % 3 == 0:
                round_count = 1
        print('-' * 50 + 'Round {} game over!'.format(this_round) + '-' * 50)

    print('winning rate: ', count / rounds)