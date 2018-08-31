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

        # some parameters for defining a current state
        self.mid_line = self.button_up_margin + self.button_height // 2
        self.max_pixel_difference = 10
        self.start_color_yellow = np.array([0, 134, 214])
        self.start_color_blue = np.array([198, 128, 0])
        self.colors = np.array([self.start_color_yellow, self.start_color_blue])
        self.chupai_start_position_yellow = 518

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
            'alone_chupai': self.alone_chupai
        }


if __name__ == '__main__':
    cf = Configuration()
    print(cf.video_path)
    print(cf.jiaodizhu[0])

