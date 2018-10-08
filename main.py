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
import datetime

current_time = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")
args_dict = {
        'kOneHandPower': [-15, -10, -5, -20],
        'kPowerUnit': [-15, -10, -5, -20],
        'RefValue': [10, 11, 12, 13, 14, 15],
        'SingleMainConstant': [0.435],
        'SingleMainCoef': [0.0151],
        'SingleMainSerialCoef': [0.02],
        'SingRemainPenalty': [0.01],
        'SingleLineExactPenalty': [-0.02],
        'DoubleMainConstant': [0.433],
        'DoubleMainCoef': [0.015],
        'DoubleMainSerialCoef': [0.02],
        'DoubleRemainPenalty': [0.01],
        'DoubleLineExactPenalty': [-0.02],
        'TripleMainConstant': [0.433],
        'TripleMainCoef': [0.02],
        'TripleMainSerialCoef': [0.02],
        'TripleMainSubCoef': [0.01],
        'TripleRemainPenalty': [0.01],
        'TripleLineExactPenalty': [-0.02],
        'QuatricWithSubConstant': [-4.5],
        'QuatricWithSubCoef': [0.003],
        'QuatricWithSubSerialCoef': [0.002],
        'QuatricWithSubSubCoef': [0.002],
        'QuatricWithoutSubConstant': [-6],
        'QuatricWithoutSubCoef': [0.175],
        'QuatricWithoutSubSerialCoef': [0.002],
        'QuatricTwoConstant': [-4.65],
        'BigBangPenalty': [-8.0]
    }


def logger(log_info, filename=current_time + '.log', foldername='pylog', verbose=True, write_log=True):
    full_name = os.path.join(foldername, filename)
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    if verbose:
        print(log_info)
    if write_log:
        with open(full_name, 'a+') as f:
            f.write(log_info)
            f.close()


def main(args, args_order, write_log_flag, remark, env, rounds=10):
    if len(args) == len(args_dict.keys()):
        count = 0
        log_info = '*************************New Parameter combinations************************\n'
        keys_list = [key for key in args_dict.keys()]
        for key_idx in range(len(keys_list)):
            log_info += 'Argument name is {}, value is {}\n'.format(keys_list[key_idx], args[key_idx])
        logger(log_info, filename=remark + current_time + '.log', write_log=write_log_flag)
        for this_round in range(rounds):
            env.reset()
            env.prepare()
            r = 0
            round_count = 1
            log_info = ''
            log_info += '-' * 50 + 'Game start at rounds {}!'.format(this_round) + '-' * 50 + '\n'
            logger(log_info, filename=remark + current_time + '.log', write_log=write_log_flag)
            while r == 0:
                intention, r, cate, idx, p1, p2, idx1, p0, idx2, my_idx = env.step_auto(args)
                # intention = to_char(intention)
                # p1 = to_char(p1)
                # p2 = to_char(p2)
                # p0 = to_char(p0)
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
            logger(log_info, filename=remark + current_time + '.log', write_log=write_log_flag)

        log_info = '***************************winning rate: {}**********************************\n'.format(count / rounds)
        logger(log_info, filename=remark + current_time + '.log', write_log=write_log_flag)
    else:
        keys_list = [key for key in args_dict.keys()]
        current_args = args_dict[keys_list[args_order]]
        for arg_value in current_args:
            args.append(arg_value)
            main(args, args_order + 1, write_log_flag, remark, env, rounds)
            if len(args) == args_order + 1:
                args.pop()


if __name__ == '__main__':
    env = CEnv()
    rounds = 500
    args = []
    write_log_flag = False
    write_log = input("Do you want to log:y(yes) / any other key(no) ?")
    remark = ''
    if write_log == 'y':
        write_log_flag = True
        remark = input("Type in some remarks for adding before the log file title: ")
    main(args, 0, write_log_flag, remark, env, rounds)