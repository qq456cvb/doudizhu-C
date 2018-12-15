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
        self.video_path = [self.video_root_path + '/' + path for path in os.listdir(self.video_root_path)]
        # self.video_path.sort(key=lambda x: int(re.findall('[0-9].*', x.split('.')[0])[0]))
        self.array_path = 'array/'

        # some global parameters
        self.channels = 3
        self.img_size = np.array([781, 1288, 3])

        # some parameters of one yellow button
        self.button_height = 74
        self.button_up_margin = 365 - 2
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
        # push a window (chuntian)
        self.push_window_left = 1132
        self.push_window_top = 74
        self.push_window_width = 48
        self.push_window_height = 33
        self.push_window_color = np.array([0, 85, 247])
        # avoid addict
        self.addict_left = 544
        self.addict_top = 155
        self.addict_width = 201
        # self.addict_color = np.array([16, 73, 173])

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
        self.fail_end = np.load(self.array_path + 'fail_end' + '.npy')
        self.addict_window = np.load(self.array_path + 'addict_window' + '.npy')
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
             # 'continous_end': self.end,
            'ming_chupai': self.ming_chupai,
            'fail_end': self.fail_end
        }

        # some parameters defining the load mark
        # self is load
        self.self_lord_x = 250
        self.self_lord_y = 488
        self.self_lord_color = np.array([115, 211, 16])
        # left player is load
        self.left_lord_x = 77
        self.left_lord_y = 285
        self.left_lord_y_with_super_mul = self.left_lord_y - 32
        self.left_lord_color = np.array([107, 195, 49])
        # right player is load
        self.right_lord_x = 1204
        self.right_lord_y = 282
        self.right_lord_y_with_super_mul = self.right_lord_y - 32
        self.right_lord_color = np.array([74, 150, 8])

        ''' x position
        jiaodizhu = 436
        bujiao = 703
        qiangdizhu = 436
        buqiang = 703
        buchu = 350
        tishi = 603
        chupai = 774
        '''


class ConfigurationOffline:
    def __init__(self):
        # path information
        self.img_root_path = 'photo/'
        self.video_root_path = 'video/'
        self.img_path = [self.img_root_path + path for path in os.listdir(self.img_root_path)]
        self.video_path = [self.video_root_path + '/' + path for path in os.listdir(self.video_root_path)]
        # self.video_path.sort(key=lambda x: int(re.findall('[0-9].*', x.split('.')[0])[0]))
        self.array_path = 'array/'

        # some global parameters
        self.channels = 3
        self.img_size = np.array([1080, 1920, 3])
        self.two_words_button_width = 141
        self.window_name = 'Genymotion for personal use - Samsung Galaxy S6 - 5.1.0 - API 22 - 1440x2560 (1920x1080, 640dpi) - 192.168.119.101'
        # self.window_names = ['Genymotion for personal use - Samsung Galaxy S6 - 5.1.0 - API 22 - 1440x2560 (1920x1080, 640dpi) - 192.168.119.101',
        #                      'Genymotion for personal use - Samsung Galaxy S6 - 5.1.0 - API 22 - 1440x2560_1 (1920x1080, 640dpi) - 192.168.119.102',
        #                      'Genymotion for personal use - Samsung Galaxy S6 - 5.1.0 - API 22 - 1440x2560_2 (1920x1080, 640dpi) - 192.168.119.102']
        self.steps_per_epoch = 50
        # self.window_name = 'BlueStacks App Player'

        # npy informations
        self.start_array = np.load(self.array_path + "start.npy")
        self.tishi_array = np.load(self.array_path + "tishi.npy")
        self.reverse_array = np.load(self.array_path + "reverse.npy")
        self.buchu_array = np.load(self.array_path + "buchu.npy")
        self.bujiao_array = np.load(self.array_path + "bujiao.npy")
        self.jiaodizhu_array = np.load(self.array_path + "jiaodizhu.npy")
        self.chupai_array = np.load(self.array_path + "chupai.npy")
        self.continuous_defeat_array = np.load(self.array_path + "continuous_defeat.npy")
        self.get_reward_array = np.load(self.array_path + "get_reward.npy")

        # some parameters defining position informations
        self.mid_line = 753
        self.button_up_margin = 723
        self.button_down_margin = 793
        # y, x, array
        self.button_information = {
            "start": [self.mid_line, 860, self.start_array],
            "tishi": [self.mid_line, 860, self.tishi_array],
            "reverse": [self.mid_line, 675, self.reverse_array],
            "buchu": [self.mid_line, 486, self.buchu_array],
            "chupai": [self.mid_line, 1240, self.chupai_array],
            "bujiao": [self.mid_line, 675, self.bujiao_array],
            "jiaodizhu": [self.mid_line, 1050, self.jiaodizhu_array],
            "continuous defeat": [200, 780, self.continuous_defeat_array],
            "get_reward": [704, 876, self.get_reward_array],
        }

        # some parameters defining the load mark
        # self is load
        self.self_lord_x = 135
        self.self_lord_y = 693
        self.self_lord_color = np.array([154, 146, 10])
        # left player is load
        self.left_lord_x = 135
        self.left_lord_y = 220
        self.left_lord_color = np.array([154, 147, 12])
        # right player is load
        self.right_lord_x = 1810
        self.right_lord_y = 220
        self.right_lord_color = np.array([154, 147, 12])

        # some parameters for judging win or lose
        self.winning_losing_top = 306
        self.winning_losing_left = 963
        self.losing_color = np.array([213, 213, 213])
        self.winning_color = np.array([13, 216, 252])

        # some parameters for restarting
        self.hanging_circle_bbox = np.array([1549, 66, 1608, 110])
        self.return_bbox = np.array([1187, 134, 1253, 193])
        self.confirm_return_bbox = np.array([1085, 532, 1144, 547])
        self.happy_battle_bbox = np.array([936, 486, 1000, 499])
        self.battle_center_bbox = np.array([1650, 114, 1706, 156])

        self.hanging__circle_pos = [(self.hanging_circle_bbox[0] + self.hanging_circle_bbox[2]) // 2,
                                    (self.hanging_circle_bbox[1] + self.hanging_circle_bbox[3]) // 2]
        self.return_pos = [(self.return_bbox[0] + self.return_bbox[2]) // 2,
                           (self.return_bbox[1] + self.return_bbox[3]) // 2]
        self.happy_battle_pos = [(self.happy_battle_bbox[0] + self.happy_battle_bbox[2]) // 2,
                                 (self.happy_battle_bbox[1] + self.happy_battle_bbox[3]) // 2]
        self.battle_center_pos = [(self.battle_center_bbox[0] + self.battle_center_bbox[2]) // 2,
                                  (self.battle_center_bbox[1] + self.battle_center_bbox[3]) // 2]
        self.confirm_return_pos = [(self.confirm_return_bbox[0] + self.confirm_return_bbox[2]) // 2,
                                   (self.confirm_return_bbox[1] + self.confirm_return_bbox[3]) // 2]

        # some parameters for monitor
        self.start_botton_pos = [80, 1000]



if __name__ == '__main__':
    import cv2

    # img[:, :, 3] = 0
    files = ['double.png']
    for f in files:
        img = cv2.imread('photo/%s' % f, cv2.IMREAD_UNCHANGED)
        img[:, :, 3] = 0
        cv2.imwrite('photo/%s' % f, img)
    # cf = Configuration()
    # print(cf.cend)
    # print(cf.jiaodizhu[0])
    # print(cf.mid_line)