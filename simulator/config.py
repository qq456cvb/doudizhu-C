#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""
import os
import re

import numpy as np


class Configuration:
    def __init__(self):
        # path information
        self.img_root_path = 'photo/'
        self.video_root_path = 'video/'
        self.img_path = [self.img_root_path + path for path in os.listdir(self.img_root_path)]
        self.video_path = [self.video_root_path + path for path in os.listdir(self.video_root_path)]
        self.video_path.sort(key=lambda x: int(re.findall('[0-9].*', x.split('.')[0])[0]))
        self.array_path = 'array/'

        # some global parameters
        self.channels = 3
        self.img_size = np.array([781, 1288, 3])

        # some parameters of one yellow button
        self.button_height = 74
        self.button_up_margin = 365
        self.button_down_margin = 439
        self.two_words_button_width = 151
        self.three_words_button_width = 152
        self.end_button_width = 243

        # some parameters for defining a current state
        self.mid_line = self.button_up_margin + self.button_height // 2
        self.end_line_y = 677
        self.end_continue_line_y = 672
        self.max_pixel_difference = 10
        self.start_color_yellow = np.array([0, 134, 214])
        self.start_color_blue = np.array([198, 128, 0])
        self.start_end_button_color = np.array([227, 160, 36])
        self.start_continue_end_button_color = np.array([217, 151, 32])
        self.colors = np.array([self.start_color_yellow, self.start_color_blue, self.start_end_button_color,
                                self.start_continue_end_button_color])
        self.chupai_start_position_yellow = 518
        self.winning_start_x = 978
        self.winning_start_y = 254
        self.winning_square = 20

        # define the actions represented by buttons
        self.jiaodizhu = np.load(self.array_path + 'jiaodizhu' + '.npy')
        self.bujiao = np.load(self.array_path + 'bujiao' + '.npy')
        self.bujiabei = np.load(self.array_path + 'bujiabei' + '.npy')
        self.buchu = np.load(self.array_path + 'buchu' + '.npy')
        self.tishi = np.load(self.array_path + 'tishi' + '.npy')
        self.chupai = np.load(self.array_path + 'chupai' + '.npy')
        self.qiangdizhu = np.load(self.array_path + 'qiangdizhu' + '.npy')
        self.buqiang = np.load(self.array_path + 'buqiang' + '.npy')
        self.yaobuqi = np.load(self.array_path + 'yaobuqi' + '.npy')
        self.alone_chupai = np.load(self.array_path + 'alone_chupai' + '.npy')
        self.end = np.load(self.array_path + 'end' + '.npy')
        self.cend = np.load(self.array_path + 'continous_end' + '.npy')
        self.ming_chupai = np.load(self.array_path + 'ming_chupai' + '.npy')
        self.actions = {
            'jiaodizhu': self.jiaodizhu,
            'bujiao': self.bujiao,
            'bujiabei': self.bujiabei,
            'buchu': self.buchu,
            'tishi': self.tishi,
            'chupai': self.chupai,
            'qiangdizhu': self.qiangdizhu,
            'buqiang': self.buqiang,
            'yaobuqi': self.yaobuqi,
            'alone_chupai': self.alone_chupai,
            'end': self.end,
             'continous_end': self.cend,
            'ming_chupai': self.ming_chupai
        }

        # some parameters defining the load mark
        # self is load
        self.self_lord_x = 250
        self.self_lord_y = 488
        self.self_lord_color = np.array([106, 196, 15])
        # left player is load
        self.left_lord_x = 78
        self.left_lord_y = 251
        self.left_lord_color = np.array([95, 157, 33])
        # right player is load
        self.right_lord_x = 1210
        self.right_lord_y = 280
        self.right_lord_color = np.array([96, 181, 114])



if __name__ == '__main__':
    cf = Configuration()
    print(cf.cend)
    print(cf.jiaodizhu[0])

