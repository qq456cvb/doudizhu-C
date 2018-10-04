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

import os


def logger(log_info, filename='log2018_10_4_19_10.log', foldername='pylog', verbose=True, write_log=True):
    full_name = os.path.join(foldername, filename)
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    if verbose:
        print(log_info)
    if write_log:
        with open(full_name, 'a+') as f:
            f.write(log_info)
            f.close()


if __name__ == '__main__':
    env = CEnv()
    rounds = 500
    count = 0
    for this_round in range(rounds):
        env.reset()
        env.prepare()
        r = 0
        round_count = 1
        log_info = ''
        log_info += '-' * 50 + 'Game start at rounds {}!'.format(this_round) + '-' * 50 + '\n'
        logger(log_info)
        while r == 0:
            intention, r, cate, idx, p1, p2, idx1, p0, idx2, my_idx = env.step_auto()
            if my_idx == 0:
                if (r != 0) and (idx == my_idx):
                    count += 1
            else:
                if (r != 0) and (idx != 0):
                    count += 1
            log_info += 'player{}: puts:{} \n'.format(idx, intention)
            log_info += 'card type is {}\n'.format(cate)
            log_info += 'my_player hand cards: {}\n'.format(p1)
            log_info += 'player{} hand cards: {}\n'.format(idx1, p2)
            log_info += 'player{} hand cards: {}\n'.format(idx2, p0)
            log_info += '-' * 50 + '\n'
            round_count += 1
            if round_count % 3 == 0:
                round_count = 1
        log_info += '-' * 50 + 'Round {} game over!'.format(this_round) + '-' * 50 + '\n'
        logger(log_info, write_log=True)

    print('winning rate: ', count / rounds)